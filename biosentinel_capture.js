/**
 * biosentinel_capture.js
 * ─────────────────────────────────────────────────────────────────────
 * Drop this script into your loan application frontend (or a Streamlit
 * components HTML block) to capture raw behavioral events.
 *
 * It records:
 *   - Keystroke timing (down_time, up_time, key)
 *   - Mouse movement, clicks, scrolls
 *   - Form field enter/exit timestamps
 *   - Paste events
 *
 * At form submission, call  BioSentinel.getPayload()  to get a JSON
 * object ready to send to the Python inference.py scorer.
 *
 * Usage
 * -----
 *   <script src="biosentinel_capture.js"></script>
 *   <script>
 *     BioSentinel.init();
 *
 *     document.getElementById("submit-btn").addEventListener("click", () => {
 *       const payload = BioSentinel.getPayload();
 *       fetch("/api/bioscore", {
 *         method: "POST",
 *         headers: { "Content-Type": "application/json" },
 *         body: JSON.stringify(payload)
 *       });
 *     });
 *   </script>
 */

const BioSentinel = (() => {

  // ── State ──────────────────────────────────────────────────────
  let keystrokes   = [];
  let mouseEvents  = [];
  let formEvents   = [];
  let pasteCount   = 0;
  let sessionStart = null;
  let activeField  = null;
  let fieldEnterTime = null;

  // Throttle mouse events — capture every 50ms max
  let lastMouseCapture = 0;
  const MOUSE_THROTTLE_MS = 50;

  // ── Keystroke listener ─────────────────────────────────────────
  const onKeyDown = (e) => {
    keystrokes.push({
      key:       e.key,
      code:      e.code,
      down_time: Date.now(),
      up_time:   null,
    });
  };

  const onKeyUp = (e) => {
    // Match the most recent unmatched keydown for this key
    for (let i = keystrokes.length - 1; i >= 0; i--) {
      if (keystrokes[i].key === e.key && keystrokes[i].up_time === null) {
        keystrokes[i].up_time = Date.now();
        break;
      }
    }
  };

  // ── Mouse listener ─────────────────────────────────────────────
  const onMouseMove = (e) => {
    const now = Date.now();
    if (now - lastMouseCapture < MOUSE_THROTTLE_MS) return;
    lastMouseCapture = now;
    mouseEvents.push({ type: "move", x: e.clientX, y: e.clientY, timestamp: now });
  };

  const onMouseDown = (e) => {
    mouseEvents.push({ type: "mousedown", x: e.clientX, y: e.clientY, timestamp: Date.now() });
  };

  const onMouseUp = (e) => {
    mouseEvents.push({ type: "mouseup", x: e.clientX, y: e.clientY, timestamp: Date.now() });
  };

  const onScroll = (e) => {
    mouseEvents.push({ type: "scroll", x: window.scrollX, y: window.scrollY, timestamp: Date.now() });
  };

  // ── Paste listener ─────────────────────────────────────────────
  const onPaste = () => { pasteCount++; };

  // ── Form field listeners ───────────────────────────────────────
  const attachFieldListeners = () => {
    // Attach to all inputs, selects, textareas
    const fields = document.querySelectorAll("input, select, textarea");
    fields.forEach((field, index) => {
      field.addEventListener("focus", () => {
        activeField    = field.name || field.id || `field_${index}`;
        fieldEnterTime = Date.now();
      });

      field.addEventListener("blur", () => {
        if (activeField !== null && fieldEnterTime !== null) {
          formEvents.push({
            field:        activeField,
            field_index:  index,
            enter_time:   fieldEnterTime,
            exit_time:    Date.now(),
          });
          activeField    = null;
          fieldEnterTime = null;
        }
      });
    });
  };

  // ── Public API ─────────────────────────────────────────────────

  const init = () => {
    sessionStart = Date.now();

    // Keyboard
    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("keyup",   onKeyUp);

    // Mouse
    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mousedown", onMouseDown);
    document.addEventListener("mouseup",   onMouseUp);
    document.addEventListener("scroll",    onScroll);

    // Paste
    document.addEventListener("paste", onPaste);

    // Form fields — attach after DOM is ready
    if (document.readyState === "complete") {
      attachFieldListeners();
    } else {
      window.addEventListener("load", attachFieldListeners);
    }

    console.log("[BioSentinel] Capture initialized.");
  };

  const getPayload = () => {
    // Close any still-active field
    if (activeField !== null && fieldEnterTime !== null) {
      formEvents.push({
        field:      activeField,
        field_index: -1,
        enter_time: fieldEnterTime,
        exit_time:  Date.now(),
      });
    }

    return {
      keystrokes:       keystrokes.filter(k => k.up_time !== null), // only complete pairs
      mouse_events:     mouseEvents,
      form_events:      formEvents,
      paste_count:      pasteCount,
      session_start_ms: sessionStart,
      session_end_ms:   Date.now(),
    };
  };

  const reset = () => {
    keystrokes   = [];
    mouseEvents  = [];
    formEvents   = [];
    pasteCount   = 0;
    sessionStart = Date.now();
    console.log("[BioSentinel] Session reset.");
  };

  const getStats = () => {
    const payload = getPayload();
    const duration = (payload.session_end_ms - payload.session_start_ms) / 1000;
    return {
      keystroke_count: payload.keystrokes.length,
      mouse_events:    payload.mouse_events.length,
      form_fields:     payload.form_events.length,
      paste_count:     payload.paste_count,
      session_sec:     duration.toFixed(1),
    };
  };

  return { init, getPayload, reset, getStats };

})();
