// The interactive front end sends requests; only backend acknowledgements and queue
// snapshots determine whether a message was accepted or delivered to the agent.
import { matchesKey } from "@earendil-works/pi-tui";

export function keyAction(data, interactive) {
	if (matchesKey(data, "ctrl+c") || matchesKey(data, "ctrl+d")) return "detach";
	if (!interactive) return null;
	if (matchesKey(data, "alt+enter")) return "follow_up";
	if (matchesKey(data, "escape")) return "abort";
	return null;
}

export class InteractiveProtocol {
	constructor(write) {
		this.write = write;
		this.state = "idle";
		this.connected = true;
		this.sequence = 0;
		this.tasks = new Set();
		this.pending = new Map();
		this.queue = [];
	}

	setTasks(items) {
		this.tasks = new Set(items.map((item) => item.value));
	}

	setState(state) {
		if (["idle", "launching", "running", "finished"].includes(state)) this.state = state;
	}

	send(type, fields = {}) {
		if (!this.connected) return { error: "The interactive backend has disconnected." };
		if (type === "prompt") {
			if (!["idle", "finished"].includes(this.state)) return { error: "A run is already active." };
			if (!this.tasks.has(fields.task)) return { error: "Choose a task supplied by the benchmark catalog." };
			if ([...this.pending.values()].some((request) => request.type === "prompt")) {
				return { error: "The task selection is awaiting the backend." };
			}
		} else if (type === "steer" || type === "follow_up") {
			if (typeof fields.message !== "string" || !fields.message.trim()) return { error: "Enter a message first." };
			if (type === "steer" && this.state !== "running") {
				return { error: "Choose a benchmark task to start a run; Enter sends a steer while it is running." };
			}
			fields = { message: fields.message.trim() };
		} else if (type === "abort") {
			if (!["launching", "running"].includes(this.state)) return { error: "There is no active run to abort." };
		} else {
			return { error: "Unknown interactive request." };
		}
		const request = { id: `ui-${++this.sequence}`, type, ...fields };
		try {
			// A false write return means stream backpressure, not rejected delivery.
			this.write(JSON.stringify(request) + "\n");
		} catch (error) {
			this.connected = false;
			return { error: `Could not send the request: ${error.message}` };
		}
		this.pending.set(request.id, request);
		return { request };
	}

	response(op) {
		const request = this.pending.get(op.id);
		this.pending.delete(op.id);
		return request;
	}

	setQueue(items) {
		this.queue = items.filter((item) => ["steer", "follow_up"].includes(item.type)
			&& ["queued", "delivered", "rejected"].includes(item.status))
			.map((item) => ({ ...item }));
		for (const item of this.queue) this.pending.delete(item.id);
	}

	messageRows() {
		const pending = [...this.pending.values()]
			.filter((request) => ["steer", "follow_up"].includes(request.type))
			.map((request) => ({ ...request, status: "awaiting acknowledgement" }));
		return [...this.queue, ...pending];
	}
}
