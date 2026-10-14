// Terminal presentation only. Runtime events, controls and values stay in the backend.
import chalk from "chalk";
import { Editor, Markdown, visibleWidth, wrapTextWithAnsi } from "@earendil-works/pi-tui";

// Match the teaser's quiet charcoal surface and readable transcript hierarchy.
export const colours = {
	background: "#111216", text: "#d6d8dd", white: "#f6f7f9", args: "#b8bdc6",
	muted: "#adb3bd", dim: "#979da8", borderMuted: "#3a3c43", ora: "#cc6254",
	accent: "#d6d8dd", success: "#7fd3a0", error: "#ed9b91", warning: "#e2c78e",
	userMsgBg: "#222329",
};
export const fg = (name, text) => chalk.hex(colours[name])(text);
export const bg = (name) => (text) => chalk.bgHex(colours[name])(text);
export const markdownTheme = {
	heading: (t) => chalk.bold(fg("white", t)), link: (t) => fg("args", t), linkUrl: (t) => fg("dim", t),
	code: (t) => fg("args", t), codeBlock: (t) => fg("args", t), codeBlockBorder: (t) => fg("dim", t),
	quote: (t) => fg("muted", t), quoteBorder: (t) => fg("dim", t), hr: (t) => fg("borderMuted", t),
	listBullet: (t) => fg("muted", t), bold: (t) => chalk.bold(t), italic: (t) => chalk.italic(t),
	underline: (t) => chalk.underline(t), strikethrough: (t) => chalk.strikethrough(t),
};
export const editorTheme = {
	borderColor: (t) => fg("borderMuted", t),
	selectList: {
		selectedPrefix: (t) => fg("white", t), selectedText: (t) => fg("white", t),
		description: (t) => fg("muted", t), scrollInfo: (t) => fg("muted", t), noMatch: (t) => fg("muted", t),
	},
};

// The teaser's exact 7×10 robot pixels, packed into half-height terminal blocks.
const robotPixels = ["....##....", "..######..", "..#.##.#..", "..######..", "....##....", ".##.##.##.", ".#......#."];
export const LOGO = robotPixels.filter((_, i) => i % 2 === 0).map((row, i) => [...row].map((upper, x) => {
	const lower = robotPixels[i * 2 + 1]?.[x];
	return upper === "#" ? lower === "#" ? "█" : "▀" : lower === "#" ? "▄" : " ";
}).join(""));

export function tokens(n) {
	if (!n) return "0";
	if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
	return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);
}

function indented(text, width, first, rest) {
	const prefixWidth = Math.max(visibleWidth(first), visibleWidth(rest));
	return wrapTextWithAnsi(text, Math.max(1, width - prefixWidth)).map((line, i) => (i ? rest : first) + line);
}

/** Keep Pi's Markdown rendering, with the teaser's dot and aligned continuation lines. */
export class TranscriptMarkdown extends Markdown {
	constructor(text, marker = "● ", colour = "text") {
		super(text, 0, 0, markdownTheme, { color: (t) => fg(colour, t) });
		this.marker = marker;
	}
	render(width) {
		return super.render(Math.max(1, width - 2)).map((line, i) => (i ? "  " : fg("white", this.marker)) + line);
	}
}

/** A call and its receipts remain one component, without a padded success-colour panel. */
export class ToolTranscript {
	constructor(op) {
		this.op = op;
		this.lines = [];
		this.error = false;
	}
	invalidate() {}
	add(lines, colour = "muted") { this.lines.push(...lines.map((text) => ({ text, colour }))); }
	finish(error) { this.error ||= error; }
	render(width) {
		const { op } = this;
		const title = chalk.bold(fg("white", op.name)) + fg("args", op.args);
		const rows = ["", ...indented(title, width, fg(this.error ? "error" : "white", "● "), "  ")];
		for (const [i, line] of op.body.entries()) {
			const comment = op.comments[i];
			const code = comment ? line.slice(0, line.length - comment.length) : line;
			rows.push(...indented(fg("args", code) + (comment ? fg("dim", comment) : ""), width, "  ", "  "));
		}
		for (const [i, { text, colour }] of this.lines.entries()) {
			rows.push(...indented(fg(colour, text), width, fg("dim", i ? "     " : "  └  "), "     "));
		}
		return rows;
	}
}

/** Prefix the real Pi editor, retaining its cursor marker, wrapping and key handling. */
export class PromptEditor extends Editor {
	render(width) {
		return super.render(Math.max(1, width - 2)).map((line, i) => {
			const prefix = i === 0 || i === this.renderedVisibleLineCount + 1 ? fg("borderMuted", "──")
				: i === 1 ? fg("white", "❯ ") : "  ";
			return prefix + line;
		});
	}
	handleMouse(event) {
		if (event.x < 2) return undefined;
		return super.handleMouse({ ...event, x: event.x - 2, width: Math.max(1, event.width - 2) });
	}
}

/** Identity lives in the wrapping header; the footer keeps the active task and real usage readable. */
export class Footer {
	state = null;
	where = "";
	invalidate() {}
	render(width) {
		const s = this.state;
		if (!s) return this.where ? wrapTextWithAnsi(fg("muted", `▸▸ ${this.where}`), width) : [];
		const context = s.context_tokens ? `context ${tokens(s.context_tokens)}${s.context_window ? ` / ${tokens(s.context_window)}` : ""}` : "";
		const pct = s.context_window ? s.context_tokens / s.context_window : 0;
		const contextStyle = pct > 0.9 ? "error" : pct > 0.7 ? "warning" : "muted";
		const top = fg("ora", "▸▸ ") + [fg("muted", this.where), fg("muted", `round ${s.round}/${s.rounds}`), fg(contextStyle, context)].filter(Boolean).join(" · ");
		const usage = [s.input_total && `↑${tokens(s.input_total)}`, s.output_total && `↓${tokens(s.output_total)}`,
			Number.isFinite(s.last_latency_s) && `last call ${Math.round(s.last_latency_s)}s`].filter(Boolean).join(" · ");
		const model = [s.model || "model unknown", s.reasoning_effort].filter(Boolean).join(" • ");
		const pad = width - visibleWidth(usage) - visibleWidth(model);
		const bottom = pad >= 2 ? [usage + " ".repeat(pad) + model] : wrapTextWithAnsi([usage, model].filter(Boolean).join(" · "), width);
		return [...wrapTextWithAnsi(top, width), ...bottom.map((line) => fg("muted", line))];
	}
}

/** Anchor controls at the terminal bottom while short transcripts leave unused room. */
export class TerminalFill {
	constructor(terminal, parts) { this.terminal = terminal; this.parts = parts; }
	invalidate() {}
	render(width) {
		const used = this.parts.reduce((height, part) => height + part.render(width).length, 0);
		return Array.from({ length: Math.max(0, this.terminal.rows - used) }, () => "");
	}
}
