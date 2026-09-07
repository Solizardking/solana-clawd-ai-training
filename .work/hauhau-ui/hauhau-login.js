/**
 * Hauhau-site Clawd wallet login (plain script — no ES modules).
 * Connect Phantom / window.solana, sign the desk SIWS challenge, mint clawd_sk_.
 * Chat goes through solgpt.us — never reads or prints the Fly operator secret.
 */
(function (root) {
  var CLAWD_MINT = "8cHzQHUS2s2h8TzCmfqPKYiM4dSt4roa3n7MyRLApump";
  var DEFAULT_DESK_URL = "https://solgpt.us";
  var DEFAULT_KEYS_URL = "https://solgpt.us/api/hauhau/keys";
  var DEFAULT_CHAT_URL = "https://solgpt.us/api/hauhau/v1/chat/completions";
  var DEFAULT_PUMP_WS_URL = "wss://clawd-ws.fly.dev/ws";
  var DEFAULT_PUMP_MCP_URL = "https://clawd-pump-mcp.fly.dev/mcp";
  var HAUHAU_MODEL = "ordlibrary/hauhau-qwen36-uncensored:IQ2_M";
  var ORB_TOKEN = "https://orbmarkets.io/token/" + CLAWD_MINT;

  function queryParam(search, name) {
    if (!search) return "";
    var match = String(search).match(new RegExp("[?&]" + name + "=([^&]+)"));
    if (!match || !match[1]) return "";
    try {
      return decodeURIComponent(match[1]);
    } catch (err) {
      return "";
    }
  }

  function keysUrlFromPage(loc, doc) {
    loc = loc || (typeof location !== "undefined" ? location : null);
    doc = doc || (typeof document !== "undefined" ? document : null);
    var fromQuery = loc ? queryParam(loc.search, "keys") : "";
    if (fromQuery) return fromQuery;
    if (doc && doc.documentElement) {
      var attr = doc.documentElement.getAttribute("data-keys-url");
      if (attr && attr.trim()) return attr.trim();
    }
    return DEFAULT_KEYS_URL;
  }

  function deskOriginFromKeysUrl(keysUrl) {
    var raw = String(keysUrl || "").replace(/\/api\/hauhau\/keys.*$/, "");
    return raw || DEFAULT_DESK_URL;
  }

  function chatUrlFromPage(loc, doc) {
    loc = loc || (typeof location !== "undefined" ? location : null);
    doc = doc || (typeof document !== "undefined" ? document : null);
    var fromQuery = loc ? queryParam(loc.search, "chat") : "";
    if (fromQuery) return fromQuery;
    if (doc && doc.documentElement) {
      var attr = doc.documentElement.getAttribute("data-chat-url");
      if (attr && attr.trim()) return attr.trim();
    }
    return deskOriginFromKeysUrl(keysUrlFromPage(loc, doc)) +
      "/api/hauhau/v1/chat/completions";
  }

  function pumpWsUrlFromPage(loc, doc) {
    loc = loc || (typeof location !== "undefined" ? location : null);
    doc = doc || (typeof document !== "undefined" ? document : null);
    var fromQuery = loc ? queryParam(loc.search, "pump") : "";
    if (fromQuery) return fromQuery;
    if (doc && doc.documentElement) {
      var attr = doc.documentElement.getAttribute("data-pump-ws-url");
      if (attr && attr.trim()) return attr.trim();
    }
    return DEFAULT_PUMP_WS_URL;
  }

  function parsePumpWsMessage(raw) {
    var msg;
    try {
      msg = typeof raw === "string" ? JSON.parse(raw) : raw;
    } catch (err) {
      return null;
    }
    if (!msg || typeof msg !== "object") return null;
    if (
      msg.type === "token-launch" ||
      msg.type === "status" ||
      msg.type === "token-enriched"
    ) {
      return msg;
    }
    return null;
  }

  function applyPumpMessage(msg, tape) {
    tape = tape || { tokens: [], status: null, enriched: {}, seen: {} };
    if (!msg || !msg.type) return tape;
    if (msg.type === "status") {
      tape.status = msg;
      return tape;
    }
    if (msg.type === "token-enriched") {
      tape.enriched = tape.enriched || {};
      if (msg.mint) tape.enriched[msg.mint] = msg;
      return tape;
    }
    var key = msg.mint || msg.signature || "";
    tape.seen = tape.seen || {};
    if (key && tape.seen[key]) return tape;
    if (key) tape.seen[key] = true;
    tape.tokens = [msg].concat(tape.tokens || []).slice(0, 40);
    return tape;
  }

  function getSolanaProvider(win) {
    win = win || (typeof window !== "undefined" ? window : null);
    if (!win) return null;
    var injected = win.solana || (win.phantom && win.phantom.solana);
    if (injected && (injected.isPhantom || typeof injected.connect === "function")) {
      return injected;
    }
    return null;
  }

  function bytesToBase64(bytes) {
    var bin = "";
    var i;
    for (i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
    return btoa(bin);
  }

  function encodeSignature(value) {
    if (!value) throw new Error("Empty wallet signature.");
    if (typeof value === "string") return value;
    if (value instanceof ArrayBuffer) return bytesToBase64(new Uint8Array(value));
    if (value.signature) return encodeSignature(value.signature);
    if (typeof value.length === "number") return bytesToBase64(value);
    throw new Error("Wallet signature was not base64 or bytes.");
  }

  function parseIssueBody(body) {
    body = body && typeof body === "object" ? body : {};
    if (typeof body.key === "string" && body.key.indexOf("clawd_sk_") === 0) {
      return {
        ok: true,
        key: body.key,
        keyPrefix:
          typeof body.keyPrefix === "string"
            ? body.keyPrefix
            : body.key.slice(0, 16),
        wallet: typeof body.wallet === "string" ? body.wallet : null,
        amount: typeof body.amount === "number" ? body.amount : null,
      };
    }
    return {
      ok: false,
      error: typeof body.error === "string" ? body.error : "issue_failed",
      message:
        typeof body.message === "string"
          ? body.message
          : "Could not issue a Hauhau key for this wallet.",
    };
  }

  function challengeUrl(keysUrl, wallet) {
    return (
      keysUrl +
      (keysUrl.indexOf("?") >= 0 ? "&" : "?") +
      "wallet=" +
      encodeURIComponent(wallet)
    );
  }

  function readPublicKey(provider, connectResult) {
    return (
      (connectResult &&
        connectResult.publicKey &&
        connectResult.publicKey.toString &&
        connectResult.publicKey.toString()) ||
      (provider &&
        provider.publicKey &&
        provider.publicKey.toString &&
        provider.publicKey.toString()) ||
      null
    );
  }

  function connectSolanaWallet(provider) {
    if (!provider || typeof provider.connect !== "function") {
      return Promise.resolve(null);
    }
    return Promise.resolve(provider.connect()).then(function (res) {
      var publicKey = readPublicKey(provider, res);
      if (!publicKey) throw new Error("Wallet did not return a Solana address.");
      return publicKey;
    });
  }

  function buildIssuePostBody(wallet, message, signature) {
    if (!wallet) throw new Error("Wallet did not return a Solana address.");
    if (!message) throw new Error("Could not build a sign-in challenge.");
    return {
      wallet: wallet,
      message: message,
      signature: encodeSignature(signature),
    };
  }

  function signAndIssue(provider, publicKey, keysUrl, fetchImpl) {
    fetchImpl =
      fetchImpl || (typeof fetch !== "undefined" ? fetch : null);
    if (!fetchImpl) {
      return Promise.reject(new Error("Could not build a sign-in challenge."));
    }
    if (!provider || typeof provider.signMessage !== "function") {
      return Promise.reject(
        new Error("This wallet cannot sign a Hauhau challenge."),
      );
    }
    return fetchImpl(challengeUrl(keysUrl, publicKey), { cache: "no-store" })
      .then(function (challengeRes) {
        return challengeRes.json();
      })
      .then(function (challenge) {
        if (
          !challenge ||
          typeof challenge.message !== "string" ||
          !challenge.message
        ) {
          throw new Error(
            (challenge && challenge.error) ||
              "Could not build a sign-in challenge.",
          );
        }
        return Promise.resolve(
          provider.signMessage(
            new TextEncoder().encode(challenge.message),
            "utf8",
          ),
        ).then(function (signed) {
          var body = buildIssuePostBody(
            publicKey,
            challenge.message,
            signed,
          );
          return fetchImpl(keysUrl, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify(body),
          }).then(function (issueRes) {
            return issueRes.json();
          }).then(function (raw) {
            return parseIssueBody(raw);
          });
        });
      });
  }

  function buildChatPostBody(messages, model) {
    if (!messages || !messages.length) {
      throw new Error("Type a message to chat with Hauhau.");
    }
    return {
      model: model || HAUHAU_MODEL,
      messages: messages,
      stream: false,
    };
  }

  function buildChatRequest(chatUrl, key, messages, model) {
    if (!key || String(key).indexOf("clawd_sk_") !== 0) {
      throw new Error("Sign in first to chat with Hauhau on solgpt.us.");
    }
    return {
      url: chatUrl || DEFAULT_CHAT_URL,
      init: {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: "Bearer " + key,
        },
        body: JSON.stringify(buildChatPostBody(messages, model)),
      },
    };
  }

  function parseChatReply(body) {
    body = body && typeof body === "object" ? body : {};
    if (body.error) {
      var err = body.error;
      var message =
        typeof body.message === "string"
          ? body.message
          : typeof err === "string"
            ? err
            : err && typeof err.message === "string"
              ? err.message
              : "Chat failed.";
      return {
        ok: false,
        error:
          typeof err === "string"
            ? err
            : (err && err.code) || "chat_failed",
        message: message,
      };
    }
    var choice = body.choices && body.choices[0];
    var content = choice && choice.message && choice.message.content;
    if (typeof content === "string" && content) {
      return { ok: true, content: content };
    }
    return {
      ok: false,
      error: "empty_reply",
      message: "Hauhau returned no text.",
    };
  }

  function sendChat(chatUrl, key, messages, fetchImpl, model) {
    fetchImpl =
      fetchImpl || (typeof fetch !== "undefined" ? fetch : null);
    if (!fetchImpl) {
      return Promise.reject(new Error("Could not reach solgpt.us chat."));
    }
    var req = buildChatRequest(chatUrl, key, messages, model);
    return fetchImpl(req.url, req.init).then(function (res) {
      return res.json().then(function (raw) {
        return parseChatReply(raw);
      });
    });
  }

  function $(id) {
    return typeof document !== "undefined" ? document.getElementById(id) : null;
  }

  function setText(id, text) {
    var el = $(id);
    if (el) el.textContent = text;
  }

  function setHidden(id, hidden) {
    var el = $(id);
    if (!el) return;
    if (hidden) el.setAttribute("hidden", "");
    else el.removeAttribute("hidden");
  }

  function setBusy(busy) {
    var btn = $("hauhau-login-btn");
    if (btn) btn.disabled = !!busy;
  }

  function shortWallet(wallet) {
    if (!wallet || wallet.length < 10) return wallet || "";
    return wallet.slice(0, 4) + "…" + wallet.slice(-4);
  }

  function appendChatLine(who, text) {
    var log = $("chat-log");
    if (!log) return;
    var wrap = document.createElement("div");
    var label = document.createElement("div");
    label.className = "who";
    label.textContent = who;
    var body = document.createElement("div");
    body.textContent = text;
    wrap.appendChild(label);
    wrap.appendChild(body);
    log.appendChild(wrap);
    log.scrollTop = log.scrollHeight;
  }

  function boot() {
    var loc = typeof location !== "undefined" ? location : { protocol: "http:" };
    var doc = typeof document !== "undefined" ? document : null;
    var keysUrl = keysUrlFromPage(loc, doc);
    var chatUrl = chatUrlFromPage(loc, doc);
    var deskUrl = deskOriginFromKeysUrl(keysUrl);
    var provider = getSolanaProvider();
    var publicKey = null;
    var sessionKey = null;
    var connecting = false;
    var chatting = false;
    var history = [];

    setHidden("file-hint", loc.protocol !== "file:");
    setText("keys-origin", deskUrl);
    var deskLink = $("open-desk-chat");
    if (deskLink) deskLink.setAttribute("href", deskUrl);
    // The launch radar has its own bounded, reconnecting stream.js controller.

    if (!provider) {
      setText(
        "wallet-status",
        "Install Phantom, then connect a Solana wallet that holds $CLAWD.",
      );
      setText("hauhau-login-btn", "Install Phantom");
    } else {
      setText("wallet-status", "Connect a Solana wallet to prove the $CLAWD holding.");
    }

    function connectWallet() {
      if (!provider) {
        if (typeof window !== "undefined") {
          window.open("https://phantom.app/", "_blank", "noopener");
        }
        return Promise.resolve(null);
      }
      connecting = true;
      setBusy(true);
      setText("hauhau-login-btn", "Connecting…");
      return connectSolanaWallet(provider)
        .then(function (pk) {
          publicKey = pk;
          if (!publicKey) throw new Error("Wallet did not return a Solana address.");
          setText("wallet-status", "Wallet " + shortWallet(publicKey));
          setText("hauhau-login-btn", "Sign in & prove $CLAWD");
          setHidden("login-error", true);
          return publicKey;
        })
        .catch(function (err) {
          setHidden("login-error", false);
          setText(
            "login-error",
            err && err.message ? err.message : "Wallet connect was cancelled.",
          );
          setText("hauhau-login-btn", "Connect Solana wallet");
          return null;
        })
        .then(function (pk) {
          connecting = false;
          setBusy(false);
          return pk;
        });
    }

    function signIn() {
      var start = publicKey
        ? Promise.resolve(publicKey)
        : connectWallet();
      return start.then(function () {
        if (!publicKey) return;
        if (!provider || typeof provider.signMessage !== "function") {
          setHidden("login-error", false);
          setText("login-error", "This wallet cannot sign a Hauhau challenge.");
          return;
        }
        setBusy(true);
        setText("hauhau-login-btn", "Signing…");
        setHidden("login-error", true);
        return signAndIssue(provider, publicKey, keysUrl)
          .then(function (issued) {
            if (!issued.ok) {
              throw new Error(issued.message);
            }
            sessionKey = issued.key;
            setHidden("login-form", true);
            setHidden("logged-in", false);
            setHidden("chat-form", false);
            setText("issued-key", issued.key);
            setText(
              "logged-in-meta",
              (issued.wallet ? shortWallet(issued.wallet) : shortWallet(publicKey)) +
                (issued.amount != null ? " · " + issued.amount + " $CLAWD" : ""),
            );
          })
          .catch(function (err) {
            setHidden("login-error", false);
            setText(
              "login-error",
              err && err.message ? err.message : "Sign-in failed.",
            );
            setText("hauhau-login-btn", "Sign in & prove $CLAWD");
          })
          .then(function () {
            setBusy(false);
          });
      });
    }

    function submitChat(text) {
      var content = String(text || "").trim();
      if (!content || chatting) return Promise.resolve(null);
      if (!sessionKey) {
        setHidden("chat-error", false);
        setText("chat-error", "Sign in first to chat with Hauhau on solgpt.us.");
        return Promise.resolve(null);
      }
      chatting = true;
      var sendBtn = $("chat-send");
      if (sendBtn) sendBtn.disabled = true;
      setHidden("chat-error", true);
      history.push({ role: "user", content: content });
      appendChatLine("You", content);
      return sendChat(chatUrl, sessionKey, history.slice())
        .then(function (reply) {
          if (!reply.ok) {
            throw new Error(reply.message || "Chat failed.");
          }
          history.push({ role: "assistant", content: reply.content });
          appendChatLine("Hauhau", reply.content);
          return reply;
        })
        .catch(function (err) {
          setHidden("chat-error", false);
          setText(
            "chat-error",
            err && err.message ? err.message : "Chat failed.",
          );
          return null;
        })
        .then(function (reply) {
          chatting = false;
          if (sendBtn) sendBtn.disabled = false;
          return reply;
        });
    }

    var btn = $("hauhau-login-btn");
    if (btn) {
      btn.addEventListener("click", function () {
        if (connecting) return;
        if (!publicKey) void connectWallet();
        else void signIn();
      });
    }

    var copyBtn = $("copy-key-btn");
    if (copyBtn) {
      copyBtn.addEventListener("click", function () {
        var keyEl = $("issued-key");
        var key = keyEl ? keyEl.textContent : "";
        if (!key || typeof navigator === "undefined" || !navigator.clipboard) return;
        void navigator.clipboard.writeText(key).then(function () {
          copyBtn.textContent = "Copied";
          setTimeout(function () {
            copyBtn.textContent = "Copy key";
          }, 2000);
        });
      });
    }

    var composer = $("chat-composer");
    if (composer) {
      composer.addEventListener("submit", function (ev) {
        if (ev && ev.preventDefault) ev.preventDefault();
        var input = $("chat-input");
        var text = input ? input.value : "";
        if (input) input.value = "";
        void submitChat(text);
      });
    }
  }

  var api = {
    CLAWD_MINT: CLAWD_MINT,
    DEFAULT_DESK_URL: DEFAULT_DESK_URL,
    DEFAULT_KEYS_URL: DEFAULT_KEYS_URL,
    DEFAULT_CHAT_URL: DEFAULT_CHAT_URL,
    DEFAULT_PUMP_WS_URL: DEFAULT_PUMP_WS_URL,
    DEFAULT_PUMP_MCP_URL: DEFAULT_PUMP_MCP_URL,
    HAUHAU_MODEL: HAUHAU_MODEL,
    ORB_TOKEN: ORB_TOKEN,
    keysUrlFromPage: keysUrlFromPage,
    chatUrlFromPage: chatUrlFromPage,
    pumpWsUrlFromPage: pumpWsUrlFromPage,
    parsePumpWsMessage: parsePumpWsMessage,
    applyPumpMessage: applyPumpMessage,
    deskOriginFromKeysUrl: deskOriginFromKeysUrl,
    getSolanaProvider: getSolanaProvider,
    bytesToBase64: bytesToBase64,
    encodeSignature: encodeSignature,
    parseIssueBody: parseIssueBody,
    challengeUrl: challengeUrl,
    connectSolanaWallet: connectSolanaWallet,
    buildIssuePostBody: buildIssuePostBody,
    signAndIssue: signAndIssue,
    buildChatPostBody: buildChatPostBody,
    buildChatRequest: buildChatRequest,
    parseChatReply: parseChatReply,
    sendChat: sendChat,
    boot: boot,
  };

  root.HauhauLogin = api;
  if (typeof document !== "undefined") {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", boot);
    } else {
      boot();
    }
  }
})(typeof window !== "undefined" ? window : this);
