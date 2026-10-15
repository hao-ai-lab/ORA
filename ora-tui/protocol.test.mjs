import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { EventEmitter } from "node:events";
import { PassThrough, Writable } from "node:stream";
import test from "node:test";
import chalk from "chalk";
import { stripTerminalSequences, visibleWidth } from "@earendil-works/pi-tui";
import { childCommand, main, parseArgs } from "./index.mjs";
import { colours, Footer, LOGO, ToolTranscript } from "./presentation.mjs";
import { InteractiveProtocol, keyAction } from "./protocol.mjs";

test("interactive keys support legacy and Kitty encodings; watch only detaches", () => {
	for (const data of ["\x1b\r", "\x1b[13;3u"]) assert.equal(keyAction(data, true), "follow_up");
	for (const data of ["\x1b", "\x1b[27u"]) assert.equal(keyAction(data, true), "abort");
	for (const data of ["\x03", "\x04", "\x1b[99;5u"]) assert.equal(keyAction(data, true), "detach");
	assert.equal(keyAction("\x1b\r", false), null);
	assert.equal(keyAction("\x1b", false), null);
	assert.equal(keyAction("\r", true), null); // Editor submits Enter; modified Enter is intercepted.
});

test("task admission is catalog-only; free text and premature steer never start a run", () => {
	const requests = [];
	const protocol = new InteractiveProtocol((raw) => requests.push(JSON.parse(raw)));
	protocol.setTasks([{ value: "benchmark/task-a" }]);
	assert.match(protocol.send("steer", { message: "Pick up a cup" }).error, /Choose a benchmark task/);
	assert.match(protocol.send("prompt", { task: "custom instruction" }).error, /catalog/);
	assert.match(protocol.send("abort").error, /no active run/);
	assert.equal(requests.length, 0);
	assert.equal(protocol.send("prompt", { task: "benchmark/task-a" }).request.type, "prompt");
	assert.match(protocol.send("prompt", { task: "benchmark/task-a" }).error, /awaiting/);
	protocol.setState("running");
	assert.match(protocol.send("prompt", { task: "benchmark/task-a" }).error, /already active/);
});

test("only backend queue snapshots mark messages queued or delivered", () => {
	const sent = [];
	const protocol = new InteractiveProtocol((raw) => sent.push(JSON.parse(raw)));
	protocol.setState("running");
	const { request } = protocol.send("steer", { message: "  Keep the left arm still.  " });
	assert.equal(sent[0].message, "Keep the left arm still.");
	assert.equal(protocol.messageRows()[0].status, "awaiting acknowledgement");
	protocol.response({ id: request.id, type: "steer", success: true });
	assert.deepEqual(protocol.messageRows(), []); // An ACK alone is not proof of model delivery.
	const item = { ...request, status: "queued" };
	protocol.setQueue([item]);
	assert.equal(protocol.messageRows()[0].status, "queued");
	item.status = "delivered";
	assert.equal(protocol.messageRows()[0].status, "queued"); // Snapshot is copied.
	protocol.setQueue([item]);
	assert.equal(protocol.messageRows()[0].status, "delivered");
});

test("follow-ups wait externally and never generate an automatic prompt", () => {
	const sent = [];
	const protocol = new InteractiveProtocol((raw) => sent.push(JSON.parse(raw)));
	assert.match(protocol.send("follow_up", { message: "  " }).error, /message/);
	protocol.send("follow_up", { message: "Inspect the scene first on the next task." });
	protocol.setState("running");
	protocol.setState("finished");
	protocol.setTasks([{ value: "task-b" }]);
	assert.deepEqual(sent.map((request) => request.type), ["follow_up"]);
});

test("closed input refuses delivery; stream backpressure preserves pending requests", () => {
	const blocked = new InteractiveProtocol(() => false);
	assert.equal(blocked.send("follow_up", { message: "hello" }).request.type, "follow_up");
	assert.equal(blocked.pending.size, 1);
	const closed = new InteractiveProtocol(() => { throw new Error("closed"); });
	assert.match(closed.send("follow_up", { message: "hello" }).error, /closed/);
	assert.match(closed.send("follow_up", { message: "again" }).error, /disconnected/);
	assert.equal(closed.pending.size, 0);
});

test("child commands keep watch read-only and forward repeated interactive specs", () => {
	assert.deepEqual(childCommand(parseArgs(["some/run", "--speed", "0"])),
		["-m", "ora.orchestrator.ora_cli", "watch", "some/run", "--emit", "jsonl", "--speed", "0"]);
	assert.deepEqual(childCommand(parseArgs(["--interactive", "--spec", "a.json", "--spec", "b.json", "--confirm-launch", "--poll", ".1"])),
		["-m", "ora.orchestrator.ora_cli", "interactive", "--spec", "a.json", "--spec", "b.json", "--confirm-launch", "--poll", ".1", "--emit", "jsonl"]);
	assert.throws(() => parseArgs(["--interactive", "--spec"]), /needs a value/);
	assert.throws(() => parseArgs(["--interactive", "--spec", "a.json", "some/run"]), /no run directory/);
	assert.throws(() => parseArgs(["some/run", "--confirm-launch"]), /usage/);
});

test("world flags reach the public backend and only camera acknowledgement displays connected", (t) => {
	const flags = ["--world-view", "auto", "--world-port", "5017", "--world-browser", "off",
		"--world-public-url", "http://127.0.0.1:5017/live"];
	assert.deepEqual(childCommand(parseArgs(["some/run", ...flags])).slice(-flags.length), flags);
	assert.deepEqual(childCommand(parseArgs(["--interactive", "--spec", "a.json", ...flags])).slice(-flags.length - 2, -2), flags);
	const app = frontEnd(t, ["some/run", ...flags]);
	app.handlers.world_view({ phase: "waiting", connected: false, message: "Waiting for the camera window",
		url: "http://127.0.0.1:5017/live" });
	let output = app.ui.render(90).map(stripTerminalSequences).join("\n");
	assert.match(output, /World · waiting/);
	assert.doesNotMatch(output, /World · connected/);
	app.handlers.world_view({ phase: "thinking", connected: true, visible: true, message: "Thinking; world paused",
		url: "http://127.0.0.1:5017/live" });
	output = app.ui.render(90).map(stripTerminalSequences).join("\n");
	assert.match(output, /World · connected · Thinking; world paused/);
	assert.match(output, /http:\/\/127\.0\.0\.1:5017\/live/);
});

class FakeTerminal {
	columns = 100;
	rows = 40;
	kittyProtocolActive = false;
	writes = [];
	stopped = 0;
	cursorShown = 0;
	start(input) { this.input = input; }
	stop() { this.stopped++; }
	write(data) { this.writes.push(data); }
	hideCursor() {}
	showCursor() { this.cursorShown++; }
	moveBy() {}
	clearLine() {}
	clearFromCursor() {}
	clearScreen() {}
	setTitle() {}
	setProgress() {}
}

function frontEnd(t, argv = ["--interactive", "--spec", "task.json"]) {
	const terminal = new FakeTerminal();
	const child = new EventEmitter();
	const sent = [];
	child.stdin = new Writable({ write(raw, _, done) { sent.push(JSON.parse(raw)); done(); } });
	child.stdout = new PassThrough();
	child.stderr = new PassThrough();
	child.kill = () => { child.killed = true; };
	child.unref = () => { child.unreferenced = true; };
	const exits = [];
	const errors = [];
	const app = main(argv, { terminal, spawn: () => child, exit: (code) => exits.push(code),
		writeError: (text) => errors.push(text) });
	t.after(() => {
		app.quit(0);
		child.stdout.end();
		child.stderr.end();
	});
	return { ...app, terminal, child, sent, exits, errors };
}

test("Pi task picker, Enter, Alt+Enter, Esc and visible queue drive the real UI", (t) => {
	const app = frontEnd(t);
	app.handlers.tasks({ items: [{ value: "task-a", label: "Task A", description: "Official benchmark instruction" }] });
	app.terminal.input("\r");
	assert.deepEqual(app.sent.map((request) => request.type), ["prompt"]);
	assert.equal(app.sent[0].task, "task-a");
	app.handlers.session_state({ state: "running", run_id: "ora-interactive-123" });
	app.editor.setText("Raise the gripper first.");
	app.terminal.input("\r");
	app.editor.setText("Inspect before the next task.");
	app.terminal.input("\x1b\r");
	app.terminal.input("\x1b");
	assert.deepEqual(app.sent.map((request) => request.type), ["prompt", "steer", "follow_up", "abort"]);
	app.handlers.queue({ items: [{ ...app.sent[1], status: "queued" }, { ...app.sent[2], status: "queued" }] });
	app.ui.renderNow(true);
	assert.match(app.terminal.writes.join(""), /steer · queued: Raise the gripper first/);
	assert.match(app.terminal.writes.join(""), /follow-up · queued: Inspect before the next task/);
	app.handlers.end({ ended: "completed" });
	assert.equal(app.terminal.stopped, 0); // Interactive end returns to a manually selected task.
	app.handlers.session_state({ state: "finished" });
	app.handlers.tasks({ items: [{ value: "task-b", label: "Task B" }] });
	assert.equal(app.sent.length, 4);
});

test("rejected editor submissions retain drafts and never become a benchmark instruction", (t) => {
	const app = frontEnd(t);
	app.editor.setText("An arbitrary task instruction");
	app.terminal.input("\r");
	assert.equal(app.sent.length, 0);
	assert.equal(app.editor.getText(), "An arbitrary task instruction");
	app.handlers.session_state({ state: "running" });
	app.terminal.input("\r");
	app.handlers.response({ ...app.sent[0], success: false, error: "Not admitted" });
	assert.equal(app.editor.getText(), "An arbitrary task instruction");
});

test("Ctrl-C detaches interactive without abort or SIGTERM; watch remains read-only", (t) => {
	const interactive = frontEnd(t);
	interactive.handlers.session_state({ state: "running" });
	interactive.terminal.input("\x03");
	assert.deepEqual(interactive.sent, []);
	assert.equal(interactive.child.killed, undefined);
	assert.equal(interactive.child.stdin.writableEnded, true);
	assert.equal(interactive.terminal.stopped, 1);
	assert.equal(interactive.terminal.cursorShown, 1);
	const watch = frontEnd(t, ["some/run"]);
	watch.editor.setText("Do not execute this");
	watch.terminal.input("\r");
	watch.terminal.input("\x1b");
	assert.deepEqual(watch.sent, []);
	watch.terminal.input("\x03");
	assert.equal(watch.child.killed, true); // Only the read-only watcher child.
});

test("backend launch errors and unexpected exits restore the terminal", (t) => {
	const failed = frontEnd(t);
	failed.child.emit("error", new Error("python is unavailable"));
	assert.equal(failed.terminal.stopped, 1);
	assert.equal(failed.terminal.cursorShown, 1);
	assert.deepEqual(failed.exits, [2]);
	assert.match(failed.errors.join(""), /python is unavailable/);
	const exited = frontEnd(t);
	exited.child.stderr.write("backend failed\n");
	exited.child.emit("exit", 3, null);
	assert.equal(exited.terminal.stopped, 1);
	assert.equal(exited.terminal.cursorShown, 1);
	assert.deepEqual(exited.exits, [3]);
	assert.match(exited.errors.join(""), /backend failed/);
});

test("real child JSONL travels in both directions and EOF detaches cleanly", async (t) => {
	const terminal = new FakeTerminal();
	const exits = [];
	let child;
	const fixture = `
		const send = (op) => process.stdout.write(JSON.stringify(op) + "\\n");
		send({op:"tasks",items:[{value:"task-a",label:"Task A"}]});
		let buffer = "";
		process.stdin.on("data", (data) => {
			buffer += data;
			let split;
			while ((split = buffer.indexOf("\\n")) >= 0) {
				const request = JSON.parse(buffer.slice(0, split));
				buffer = buffer.slice(split + 1);
				if (request.type === "prompt") send({op:"session_state",state:"running",run_id:"fixture"});
				if (request.type === "steer") send({op:"queue",items:[{...request,status:"queued"}]});
				send({op:"response",...request,success:true});
			}
		});
		process.stdin.on("end", () => process.exit(0));
	`;
	const app = main(["--interactive", "--spec", "task.json"], {
		terminal, exit: (code) => exits.push(code), writeError: (text) => assert.fail(text),
		spawn: () => (child = spawn(process.execPath, ["-e", fixture], { stdio: ["pipe", "pipe", "pipe"] })),
	});
	t.after(() => { app.quit(0); if (child.exitCode === null) child.kill(); });
	const until = async (condition) => {
		const deadline = Date.now() + 2000;
		while (!condition()) {
			if (Date.now() > deadline) assert.fail("fixture backend did not respond");
			await new Promise((resolve) => setTimeout(resolve, 5));
		}
	};
	await until(() => app.protocol.tasks.size === 1);
	terminal.input("\r");
	await until(() => app.protocol.state === "running");
	app.editor.setText("Stop using the left arm.");
	terminal.input("\r");
	await until(() => app.protocol.queue.length === 1);
	assert.equal(app.protocol.queue[0].message, "Stop using the left arm.");
	terminal.input("\x03");
	await until(() => child.exitCode !== null);
	assert.equal(child.exitCode, 0);
	assert.equal(child.killed, false);
	assert.deepEqual(exits, [0]);
	assert.equal(terminal.stopped, 1);
});

test("compact tools preserve code, results and receipts without success background panels", () => {
	const previousLevel = chalk.level;
	chalk.level = 3;
	try {
		const tool = new ToolTranscript({ name: "move_pose", args: `(target="${"pose-".repeat(30)}")`,
			body: ["z = 0.7256  # measured support height"], comments: ["# measured support height"] });
		tool.add(["requested: xyz=[-0.125, 0.025, 0.8179]m", "receipt  completed · observed xyz=[-0.125, 0.025, 0.8178]m"]);
		tool.finish(false);
		for (const width of [90, 110]) {
			const rows = tool.render(width);
			assert.ok(rows.every((row) => visibleWidth(row) <= width));
			const output = rows.map(stripTerminalSequences).join("\n");
			assert.match(output, /● move_pose/);
			assert.match(output, /# measured support height/);
			assert.match(output, /└  requested:/);
			assert.match(output, /receipt  completed/);
			assert.doesNotMatch(rows.join(""), /\x1b\[48;/); // Tools do not claim execution success through a large green surface.
			assert.match(rows.join(""), /38;2;173;179;189m/); // Readable #adb3bd tool output.
		}
		tool.finish(true);
		tool.finish(false); // A later receipt must not erase an earlier tool error.
		assert.match(tool.render(90).join(""), /38;2;237;155;145m●/);
	} finally { chalk.level = previousLevel; }
});

test("footer preserves task, model, usage and actual context counts at recording widths", () => {
	const footer = new Footer();
	footer.where = "RoboTwin / place_shoe · replay ×10";
	footer.state = { round: 1, rounds: 2, input_total: 257100, output_total: 21900, context_tokens: 68700,
		context_window: 1000000, last_latency_s: 21.7, model: "DeepSeek-V4.1-Flash", reasoning_effort: "low" };
	for (const width of [90, 110]) {
		const rows = footer.render(width);
		assert.equal(rows.length, 2);
		assert.ok(rows.every((row) => visibleWidth(row) <= width));
		const output = rows.map(stripTerminalSequences).join("\n");
		for (const value of ["RoboTwin / place_shoe", "replay ×10", "round 1/2", "context 68.7k / 1.0M",
			"↑257.1k", "↓21.9k", "last call 22s", "DeepSeek-V4.1-Flash • low"]) assert.ok(output.includes(value), value);
	}
});

test("Pi rendering anchors the visible prompt and keeps complete identity at 90×44 and 110×44", (t) => {
	const app = frontEnd(t, ["some/run"]);
	const runId = "rpent-zeroshot-hdev-robotwin-a10efe025260--20261005T124118Z";
	app.handlers.header({ version: "v0.14.0", mode: "replay ×10", run_id: runId,
		benchmark: "RoboTwin", task: "place_shoe", experiment: "rpent-zeroshot-hdev",
		model: "DeepSeek-V4.1-Flash", reasoning_effort: "low", provider_protocol: "openai_chat_completions" });
	app.handlers.user({ text: "Use one arm to grab the shoe from the table and place it on the mat." });
	app.handlers.assistant({ text: "I will inspect the scene before moving." });
	app.handlers.footer({ round: 1, rounds: 1, context_tokens: 28200, context_window: 1000000,
		input_total: 0, output_total: 0, last_latency_s: null, model: "DeepSeek-V4.1-Flash", reasoning_effort: "low" });
	for (const width of [90, 110]) {
		app.terminal.columns = width;
		app.terminal.rows = 44;
		const rows = app.ui.render(width);
		assert.equal(rows.length, 44);
		assert.ok(rows.every((row) => visibleWidth(row) <= width));
		const plain = rows.map(stripTerminalSequences);
		assert.ok(plain.join("").replace(/\s/g, "").includes(runId));
		assert.match(plain.join("\n"), /❯ Use one arm/);
		assert.match(plain.join("\n"), /● I will inspect/);
		assert.match(plain.at(-4), /❯/); // Last four rows: prompt, rule, compact footer.
		assert.ok(!plain.slice(-2).join("").includes(runId));
	}
	assert.deepEqual(LOGO, [" ▄█▀▀█▄      ", " ▀█▄▄█▀▀▀▀██ ", "   ██    ▄██▄", "  ▄██▄   █  █", " ▀▀▀▀▀▀      "]);
	assert.equal(colours.background, "#111216");
	assert.equal(colours.ora, "#4ecdc4"); // Hao AI Lab theme colour on dark; distinct from the verdict green.
	assert.notEqual(colours.ora, colours.success);
});

test("prefixed Pi editor retains wrapped Unicode drafts, cursor geometry and submission", (t) => {
	const app = frontEnd(t);
	app.handlers.session_state({ state: "running" });
	const text = "先观察，再移动。" + "Keep the shoe level. ".repeat(12);
	for (const width of [90, 110]) {
		app.terminal.columns = width;
		app.terminal.rows = 44;
		app.editor.setText(text);
		const rows = app.editor.render(width);
		assert.ok(rows.every((row) => visibleWidth(row) <= width));
		assert.match(stripTerminalSequences(rows[1]), /^❯ /);
		assert.equal(app.editor.lastWidth, width - 3); // Prefix plus Pi's reserved cursor column.
		assert.equal(app.editor.getText(), text);
	}
	app.terminal.input("\r");
	assert.equal(app.sent[0].type, "steer");
	assert.equal(app.sent[0].message, text.trim());
});

test("picker header shows the prepared model and a preflight's checks stay one row each", (t) => {
	const app = frontEnd(t);
	app.handlers.header({ version: "v0.14.0", mode: "interactive", run_id: "choose a task", benchmark: "", task: "",
		experiment: "ora-interactive", model: "DeepSeek-V4.1-Flash", reasoning_effort: "low",
		provider_protocol: "openai_compatible" });
	app.handlers.session_state({ state: "launching", message: "Preflight: robotwin / click_bell" });
	for (const text of ["[PASS] source_clean", "[PASS] host_marker", "[FAIL] service_ready: {'error': 'down'}"]) {
		app.handlers.notice({ level: "info", mark: "·", text });
	}
	const plain = app.ui.render(100).map(stripTerminalSequences).map((row) => row.trimEnd());
	const output = plain.join("\n");
	assert.match(output, /DeepSeek-V4\.1-Flash · low effort · openai_compatible/);
	assert.doesNotMatch(output, /model unknown/);
	const first = plain.findIndex((row) => row.includes("[PASS] source_clean"));
	assert.ok(first > 0 && plain[first - 1].trim() === ""); // the block is separated once
	assert.match(plain[first + 1], /\[PASS\] host_marker/);
	assert.match(plain[first + 2], /\[FAIL\] service_ready: \{'error': 'down'\}/);
});
