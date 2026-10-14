// One bounded request/decode pipeline; an old binding can never replace a new one.
(() => {
  const byId = id => document.getElementById(id);
  let binding = null, decodedKey = null, generation = 0;
  const timeoutMs = 4000;
  async function fetchTimed(url, options = {}, body = null) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url, {...options, signal: controller.signal, cache: 'no-store'});
      if (!response.ok) throw new Error(`Viewer response ${response.status}`);
      if (body === 'json') return await response.json();
      if (body === 'blob') return await response.blob();
      return response;
    } finally { clearTimeout(timer); }
  }
  async function decodeCamera(camera) {
    const objectUrl = URL.createObjectURL(await fetchTimed(camera.url, {}, 'blob'));
    const image = new Image();
    image.alt = camera.name;
    image.src = objectUrl;
    let timer;
    try {
      await Promise.race([image.decode(), new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error('Camera decode timed out')), timeoutMs);
      })]);
      const figure = document.createElement('figure');
      const caption = document.createElement('figcaption');
      caption.textContent = camera.name;
      figure.append(image, caption);
      return {figure, objectUrl};
    } catch (error) { URL.revokeObjectURL(objectUrl); throw error; }
    finally { clearTimeout(timer); }
  }
  let shown = [];
  async function poll() {
    try {
      const state = await fetchTimed('/api/live/status', {}, 'json');
      if (binding !== state.binding_id) {
        binding = state.binding_id; generation++; decodedKey = null;
        byId('world-cameras').replaceChildren();
        shown.forEach(url => URL.revokeObjectURL(url)); shown = [];
        byId('world-frame').textContent = '';
      }
      const expectedGeneration = generation;
      byId('world-state').textContent = state.message;
      byId('world-identity').textContent = state.run_id || 'No task selected';
      byId('world-connection').textContent = state.frame ? 'Viewer reachable · waiting for camera decode' : 'Viewer reachable · waiting for original pixels';
      if (state.frame) {
        const frame = state.frame;
        const key = `${binding}:${frame.index}:${frame.hash}`;
        if (decodedKey !== key) {
          const results = await Promise.allSettled(frame.cameras.map(decodeCamera));
          const cameras = results.filter(result => result.status === 'fulfilled').map(result => result.value);
          if (results.some(result => result.status === 'rejected')) {
            cameras.forEach(camera => URL.revokeObjectURL(camera.objectUrl));
            throw new Error('Original camera frame could not be decoded');
          }
          // Recheck the current binding after decoding: the task may have changed.
          let current;
          try {
            current = await fetchTimed('/api/live/status', {}, 'json');
          } catch (error) {
            cameras.forEach(camera => URL.revokeObjectURL(camera.objectUrl));
            throw error;
          }
          if (generation !== expectedGeneration || current.binding_id !== binding) {
            cameras.forEach(camera => URL.revokeObjectURL(camera.objectUrl));
            return;
          }
          byId('world-cameras').replaceChildren(...cameras.map(camera => camera.figure));
          shown.forEach(url => URL.revokeObjectURL(url)); shown = cameras.map(camera => camera.objectUrl);
          await fetchTimed('/api/live/ack', {method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({binding_id: binding, ack_token: state.ack_token, frame_index: frame.index, frame_hash: frame.hash})});
          decodedKey = key;
          if (current.frame?.index !== frame.index) {
            byId('world-state').textContent = 'Captured execution frame · catching up to the current world state.';
          } else {
            byId('world-state').textContent = current.message;
          }
        }
        const age = frame.age_s == null ? 'source clock differs' : `captured ${frame.age_s.toFixed(1)}s ago`;
        byId('world-frame').textContent = `Round ${frame.round_index + 1} · simulation ${frame.sim_time_s.toFixed(3)}s · ${age} · frame ${frame.capture_index}`;
        byId('world-connection').textContent = 'Original camera frame decoded';
      }
    } catch (error) {
      byId('world-connection').textContent = `Viewer disconnected or frame unavailable · ${error.message}`;
    } finally {
      setTimeout(poll, 200); // Never overlap fetches or camera decodes.
    }
  }
  poll();
})();
