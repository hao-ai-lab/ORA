#!/usr/bin/env node
// ORA's terminal front end. It draws the UI operations `ora watch --emit jsonl` writes, with
// pi-tui (the library behind the Pi coding agent), with ORA's compact teaser-style
// transcript: header, user prompt, assistant/tool dots and indented results, spinner,
// editor and a two-line footer. The Python child owns
// every number shown. Watch is read-only; interactive mode sends explicit requests to
// the orchestrator backend, which owns task admission and operator-message delivery.
//
//   node ora-tui/index.mjs RUN [--python PY] [--speed X] [--max-gap S] [--poll S]
//   node ora-tui/index.mjs --interactive --spec PATH [--spec PATH...] [--confirm-launch]

import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { pathToFileURL } from "node:url";
import chalk from "chalk";
import {
	Box, Container, Loader, ProcessTerminal, SelectList, Spacer, Text, TuiMainScreen, matchesKey,
} from "@earendil-works/pi-tui";
import { InteractiveProtocol, keyAction } from "./protocol.mjs";

import {
	bg, editorTheme, fg, Footer, LOGO, PromptEditor, TerminalFill, ToolTranscript, TranscriptMarkdown,
} from "./presentation.mjs";

export function parseArgs(argv) {
	const args = { run: null, python: process.env.ORA_PYTHON || "python3", pass: [], interactive: false, specs: [] };
	for (let i = 0; i < argv.length; i++) {
		const a = argv[i];
		if (a === "--interactive") args.interactive = true;
		else if (a === "--confirm-launch") args.pass.push(a);
		else if (["--python", "--spec", "--speed", "--max-gap", "--poll",
			"--world-view", "--world-port", "--world-browser", "--world-public-url"].includes(a)) {
			const value = argv[++i];
			if (!value || value.startsWith("--")) throw new Error(`${a} needs a value`);
			if (a === "--python") args.python = value;
			else if (a === "--spec") args.specs.push(value);
			else args.pass.push(a, value);
		} else if (!a.startsWith("-") && !args.run) args.run = a;
		else throw new Error(`unknown argument: ${a}`);
	}
	if (args.interactive) {
		if (args.run || !args.specs.length) throw new Error("interactive mode needs --spec PATH and no run directory");
		if (args.pass.some((value) => ["--speed", "--max-gap"].includes(value))) {
			throw new Error("replay options apply to watch only");
		}
	} else if (!args.run || args.specs.length || args.pass.includes("--confirm-launch")) {
		throw new Error("usage: node ora-tui/index.mjs RUN [--python PY] [--speed X] [--max-gap S] [--poll S]");
	}
	return args;
}

export function childCommand(args) {
	return args.interactive
		? ["-m", "ora.orchestrator.ora_cli", "interactive",
			...args.specs.flatMap((path) => ["--spec", path]), ...args.pass, "--emit", "jsonl"]
		: ["-m", "ora.orchestrator.ora_cli", "watch", args.run, "--emit", "jsonl", ...args.pass];
}

export function main(argv = process.argv.slice(2), dependencies = {}) {
	const args = parseArgs(argv);
	const exit = dependencies.exit || ((code) => process.exit(code));
	const writeError = dependencies.writeError || ((text) => process.stderr.write(text));
	const ui = new TuiMainScreen(dependencies.terminal || new ProcessTerminal());
	const header = new Container();
	const chat = new Container();
	const picker = new Container();
	const status = new Container();
	const messages = new Container();
	const world = new Text("", 1, 0);
	const editor = new PromptEditor(ui, editorTheme);
	const help = new Text("", 1, 0);
	const footer = new Footer();
	const parts = [header, chat, picker, status, messages, world, editor, help, footer];
	const stage = new Box(1, 0, bg("background"));
	for (const part of [header, chat, new TerminalFill(ui.terminal, parts), picker, status, messages, world, editor, help, footer]) stage.addChild(part);
	ui.addChild(stage);
	ui.setFocus(editor);

	const tools = new Map();
	let loader = null;
	let replay = false;
	let activeTask = "ORA interactive";
	let taskList = null;
	let pickerFocused = false;
	let noticeBlock = false; // consecutive notices (a preflight's checks) share one block
	const add = (component) => {
		chat.addChild(component);
		noticeBlock = false;
		ui.requestRender();
	};
	const line = (text) => add(new Text(text, 1, 0));
	const child = (dependencies.spawn || spawn)(args.python, childCommand(args), {
		stdio: [args.interactive ? "pipe" : "ignore", "pipe", "pipe"],
		env: { ...process.env, ORA_WORLD_UI: "1" },
	});
	const protocol = args.interactive ? new InteractiveProtocol((raw) => {
		if (!child.stdin.writable || child.stdin.destroyed) throw new Error("backend input is closed");
		child.stdin.write(raw);
	}) : null;
	const showHelp = () => {
		if (!protocol) return;
		const text = pickerFocused ? "↑/↓ choose benchmark task · Enter start · Tab message editor · Ctrl-C detach"
			: protocol.state === "running" ? "Enter steer · Alt+Enter queue follow-up · Esc abort · Ctrl-C detach"
				: protocol.state === "launching" ? "Launching · Alt+Enter queue follow-up · Esc abort · Ctrl-C detach"
					: "Tab choose benchmark task · Alt+Enter queue for the next selected task · Ctrl-C detach";
		help.setText(fg("dim", text));
	};
	const drawMessages = () => {
		messages.clear();
		const rows = protocol.messageRows();
		if (rows.length) {
			messages.addChild(new Spacer(1));
			messages.addChild(new Text(fg("dim", "Operator messages"), 1, 0));
			for (const row of rows) {
				const kind = row.type === "follow_up" ? "follow-up" : "steer";
				const colour = row.status === "rejected" ? "error" : row.status === "delivered" ? "success" : "muted";
				messages.addChild(new Text(fg(colour, `${kind} · ${row.status}: ${row.message}${row.error ? ` (${row.error})` : ""}`), 1, 0));
			}
		}
		ui.requestRender();
	};
	const submit = (type, text) => {
		const result = protocol.send(type, { message: text });
		if (result.error) {
			line(fg("warning", result.error));
			if (text && !editor.getText()) editor.setText(text);
		} else {
			editor.addToHistory(text);
			editor.setText("");
			drawMessages();
		}
	};
	editor.onSubmit = (text) => {
		if (protocol) submit("steer", text);
		else {
			editor.setText("");
			chat.addChild(new Spacer(1));
			line(fg("warning", "read-only: ora watch cannot send an instruction to a run yet"));
		}
	};
	let stderr = "";
	child.stderr.on("data", (d) => (stderr += d));
	let quitting = false;
	const quit = (code) => {
		if (quitting) return;
		quitting = true;
		loader?.stop();
		protocol && (protocol.connected = false);
		if (protocol) {
			// EOF detaches the controller. Abort is a separate explicit request.
			child.stdin.end();
			child.unref?.();
		} else child.kill("SIGTERM");
		ui.stop();
		process.removeListener("SIGTERM", onSignal);
		process.removeListener("SIGINT", onSignal);
		if (stderr.trim() && code) writeError(stderr);
		exit(code);
	};
	const onSignal = () => quit(0);
	process.once("SIGTERM", onSignal);
	process.once("SIGINT", onSignal);
	ui.addInputListener((data) => {
		const action = keyAction(data, args.interactive);
		if (action === "detach") {
			quit(0); // Ctrl-C / Ctrl-D stop watching, not the run
			return { consume: true };
		}
		if (action === "follow_up") {
			submit("follow_up", editor.getExpandedText());
			return { consume: true };
		}
		if (action === "abort") {
			if (["launching", "running"].includes(protocol.state)) {
				const result = protocol.send("abort");
				if (result.error) line(fg("warning", result.error));
			} else if (pickerFocused) {
				pickerFocused = false;
				ui.setFocus(editor);
				showHelp();
			}
			return { consume: true };
		}
		if (protocol && taskList && matchesKey(data, "tab") && ["idle", "finished"].includes(protocol.state)) {
			pickerFocused = !pickerFocused;
			ui.setFocus(pickerFocused ? taskList : editor);
			showHelp();
			return { consume: true };
		}
		return undefined;
	});

	const handlers = {
		world_view(op) {
			const connection = op.connected ? "connected" : op.phase === "unavailable" ? "unavailable"
				: op.phase === "replay" ? "replay" : "waiting";
			const detail = op.connected ? op.message || op.phase : op.message || "Waiting for camera window";
			const label = ["World", connection, detail, op.url, op.browser_message].filter(Boolean).join(" · ");
			world.setText(fg(op.phase === "unavailable" ? "warning" : "muted", label));
			ui.requestRender();
		},
		header(op) {
			replay = String(op.mode).startsWith("replay");
			if (protocol) header.clear();
			const info = [
				chalk.bold(fg("ora", "ORA")) + fg("muted", ` ${op.version}`) + fg("dim", "  open robot-use agent"),
				fg("dim", [op.model || "model unknown", op.reasoning_effort && `${op.reasoning_effort} effort`, op.provider_protocol].filter(Boolean).join(" · ")),
				fg("dim", [op.task ? `${op.benchmark} / ${op.task}` : op.benchmark, op.experiment].filter(Boolean).join(" · ")),
				fg("dim", `${op.run_id} · ${op.mode}`),
			];
			header.addChild(new Text(LOGO.map((l, i) => fg("ora", l) + "  " + (info[i] ?? "")).join("\n"), 1, 1));
			activeTask = [op.benchmark, op.task].filter(Boolean).join(" / ") || "ORA";
			footer.where = `${activeTask} · ${op.mode}`;
			ui.requestRender();
		},
		tasks(op) {
			if (!protocol) return;
			const items = Array.isArray(op.items) ? op.items.filter((item) => typeof item.value === "string") : [];
			protocol.setTasks(items);
			picker.clear();
			taskList = null;
			pickerFocused = false;
			picker.addChild(new Spacer(1));
			picker.addChild(new Text(op.message || "Choose a benchmark task", 1, 0));
			if (items.length) {
				taskList = new SelectList(items, 8, editorTheme.selectList);
				taskList.onSelect = (item) => {
					const result = protocol.send("prompt", { task: item.value });
					if (result.error) line(fg("warning", result.error));
				};
				taskList.onCancel = () => {
					pickerFocused = false;
					ui.setFocus(editor);
					showHelp();
				};
				picker.addChild(taskList);
				pickerFocused = true;
				ui.setFocus(taskList);
			} else ui.setFocus(editor);
			showHelp();
			ui.requestRender();
		},
		session_state(op) {
			if (!protocol) return;
			protocol.setState(op.state);
			if (["launching", "running"].includes(protocol.state)) {
				picker.clear();
				taskList = null;
				pickerFocused = false;
				ui.setFocus(editor);
			}
			if (op.message) line(fg("dim", op.message));
			footer.where = `${activeTask} · ${protocol.state}`;
			showHelp();
			ui.requestRender();
		},
		queue(op) {
			if (!protocol) return;
			protocol.setQueue(Array.isArray(op.items) ? op.items : []);
			drawMessages();
		},
		response(op) {
			if (!protocol) return;
			const request = protocol.response(op);
			if (!op.success) {
				const error = typeof op.error === "string" ? op.error : JSON.stringify(op.error || "Request rejected");
				line(fg("error", error));
				if (request?.message && !editor.getText()) editor.setText(request.message);
			} else if (op.type === "abort") line(fg("dim", "Abort requested; waiting for the active run to stop."));
			drawMessages();
		},
		user(op) {
			if (op.operator) chat.addChild(new Spacer(1)); // an interjection is a new block, like an assistant turn
			const box = new Box(0, 0, bg("userMsgBg"));
			box.addChild(new TranscriptMarkdown(op.text, "❯ ", "white"));
			add(box);
		},
		assistant(op) {
			chat.addChild(new Spacer(1));
			add(new TranscriptMarkdown(op.text));
		},
		tool_call(op) {
			const box = new ToolTranscript(op);
			tools.set(op.id, box);
			add(box);
		},
		tool_result(op) {
			const box = tools.get(op.id);
			if (!box) return;
			const lines = op.of && op.lines.length ? [`${op.of}: ${op.lines[0]}`, ...op.lines.slice(1)] : op.lines;
			box.add(lines, op.error ? "error" : "muted");
			if (op.done) box.finish(op.error);
		},
		tool_images(op) {
			tools.get(op.id)?.add([`${op.count} image${op.count > 1 ? "s" : ""} to the model`], "dim");
		},
		receipt(op) {
			const box = tools.get(op.id);
			if (!box) return;
			box.add(op.lines.map((l) => `receipt  ${l}`));
			box.finish(false);
		},
		notice(op) {
			if (!noticeBlock) chat.addChild(new Spacer(1));
			add(new Text(fg(op.level === "warning" ? "warning" : "dim", `${op.mark} ${op.text}`), 1, 0));
			noticeBlock = true;
		},
		verdict(op) {
			const label = op.rounds > 1 ? `Round ${op.round}/${op.rounds} · ` : "";
			const verdict = op.success === true ? fg("success", "success") : op.success === false ? "failure" : "none";
			chat.addChild(new Spacer(1));
			add(new Text(`${op.success ? fg("success", "●") : "●"} ${label}official verdict (${op.benchmark} evaluator): ${verdict}`, 1, 0));
		},
		rule(op) {
			chat.addChild(new Spacer(1));
			add(new Text(fg("dim", `── ${op.text} ──`), 1, 0));
		},
		status(op) {
			if (!op.text) {
				loader?.stop();
				loader = null;
				status.clear();
			} else if (loader) {
				loader.setMessage(op.text);
			} else {
				loader = new Loader(ui, (s) => fg("ora", s), (t) => fg("muted", t), op.text, {
					frames: [..."·✢✳✶✻✽✻✶✳✢"].map((s) => fg("ora", s)), intervalMs: 110,
				});
				status.clear();
				status.addChild(loader);
			}
			ui.requestRender();
		},
		footer(op) {
			footer.state = op;
			ui.requestRender();
		},
		end() {
			if (!protocol) setTimeout(() => quit(0), replay ? 1500 : 200);
		},
	};

	createInterface({ input: child.stdout }).on("line", (raw) => {
		let op;
		try {
			op = JSON.parse(raw);
		} catch {
			return;
		}
		handlers[op.op]?.(op);
	});
	child.on("error", (error) => {
		stderr += `ora: could not start the backend: ${error.message}\n`;
		quit(2);
	});
	child.stdin?.on("error", (error) => {
		if (quitting) return;
		stderr += `ora: backend input closed: ${error.message}\n`;
		quit(2);
	});
	child.on("exit", (code, signal) => {
		if (protocol || code || signal) quit(code || (signal ? 1 : 0));
	});
	showHelp();
	ui.start();
	return { ui, editor, protocol, handlers, quit };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
	try {
		main();
	} catch (error) {
		process.stderr.write(`ora: ${error.message}\n`);
		process.exit(2);
	}
}
