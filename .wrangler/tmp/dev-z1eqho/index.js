var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });

// .wrangler/tmp/bundle-OzcOnk/checked-fetch.js
var urls = /* @__PURE__ */ new Set();
function checkURL(request, init) {
  const url = request instanceof URL ? request : new URL(
    (typeof request === "string" ? new Request(request, init) : request).url
  );
  if (url.port && url.port !== "443" && url.protocol === "https:") {
    if (!urls.has(url.toString())) {
      urls.add(url.toString());
      console.warn(
        `WARNING: known issue with \`fetch()\` requests to custom HTTPS ports in published Workers:
 - ${url.toString()} - the custom port will be ignored when the Worker is published using the \`wrangler deploy\` command.
`
      );
    }
  }
}
__name(checkURL, "checkURL");
globalThis.fetch = new Proxy(globalThis.fetch, {
  apply(target, thisArg, argArray) {
    const [request, init] = argArray;
    checkURL(request, init);
    return Reflect.apply(target, thisArg, argArray);
  }
});

// .wrangler/tmp/bundle-OzcOnk/strip-cf-connecting-ip-header.js
function stripCfConnectingIPHeader(input, init) {
  const request = new Request(input, init);
  request.headers.delete("CF-Connecting-IP");
  return request;
}
__name(stripCfConnectingIPHeader, "stripCfConnectingIPHeader");
globalThis.fetch = new Proxy(globalThis.fetch, {
  apply(target, thisArg, argArray) {
    return Reflect.apply(target, thisArg, [
      stripCfConnectingIPHeader.apply(null, argArray)
    ]);
  }
});

// node_modules/itty-router/index.mjs
var e = /* @__PURE__ */ __name(({ base: e2 = "", routes: t = [], ...o2 } = {}) => ({ __proto__: new Proxy({}, { get: (o3, s2, r, n) => "handle" == s2 ? r.fetch : (o4, ...a) => t.push([s2.toUpperCase?.(), RegExp(`^${(n = (e2 + o4).replace(/\/+(\/|$)/g, "$1")).replace(/(\/?\.?):(\w+)\+/g, "($1(?<$2>*))").replace(/(\/?\.?):(\w+)/g, "($1(?<$2>[^$1/]+?))").replace(/\./g, "\\.").replace(/(\/?)\*/g, "($1.*)?")}/*$`), a, n]) && r }), routes: t, ...o2, async fetch(e3, ...o3) {
  let s2, r, n = new URL(e3.url), a = e3.query = { __proto__: null };
  for (let [e4, t2] of n.searchParams)
    a[e4] = a[e4] ? [].concat(a[e4], t2) : t2;
  for (let [a2, c2, i2, l2] of t)
    if ((a2 == e3.method || "ALL" == a2) && (r = n.pathname.match(c2))) {
      e3.params = r.groups || {}, e3.route = l2;
      for (let t2 of i2)
        if (null != (s2 = await t2(e3.proxy ?? e3, ...o3)))
          return s2;
    }
} }), "e");
var o = /* @__PURE__ */ __name((e2 = "text/plain; charset=utf-8", t) => (o2, { headers: s2 = {}, ...r } = {}) => void 0 === o2 || "Response" === o2?.constructor.name ? o2 : new Response(t ? t(o2) : o2, { headers: { "content-type": e2, ...s2.entries ? Object.fromEntries(s2) : s2 }, ...r }), "o");
var s = o("application/json; charset=utf-8", JSON.stringify);
var c = o("text/plain; charset=utf-8", String);
var i = o("text/html");
var l = o("image/jpeg");
var p = o("image/png");
var d = o("image/webp");

// packages/worker/dist/handlers/predictions.js
async function handlePredictions(request, env) {
  const url = new URL(request.url);
  const raceId = url.pathname.split("/").pop();
  try {
    const { results } = await env.DB.prepare(`SELECT 
        p.*,
        d.name as driver_name,
        d.code as driver_code,
        d.team as driver_team,
        r.name as race_name,
        r.date as race_date
       FROM predictions p
       JOIN drivers d ON p.driver_id = d.id
       JOIN races r ON p.race_id = r.id
       WHERE p.race_id = ?
       ORDER BY p.predicted_position`).bind(raceId).all();
    return new Response(JSON.stringify({ predictions: results }), {
      headers: { "Content-Type": "application/json" }
    });
  } catch (error) {
    return new Response(JSON.stringify({ error: "Failed to fetch predictions" }), {
      status: 500,
      headers: { "Content-Type": "application/json" }
    });
  }
}
__name(handlePredictions, "handlePredictions");
async function handleLatestPredictions(request, env) {
  try {
    const { results: races } = await env.DB.prepare(`SELECT id FROM races 
       WHERE date >= date('now') 
       ORDER BY date ASC 
       LIMIT 1`).all();
    if (races.length === 0) {
      const { results: pastRaces } = await env.DB.prepare(`SELECT id FROM races 
         ORDER BY date DESC 
         LIMIT 1`).all();
      if (pastRaces.length === 0) {
        return new Response(JSON.stringify({ predictions: [] }), {
          headers: { "Content-Type": "application/json" }
        });
      }
      races[0] = pastRaces[0];
    }
    const raceId = races[0].id;
    const { results: predictions } = await env.DB.prepare(`SELECT 
        p.*,
        d.name as driver_name,
        d.code as driver_code,
        d.team as driver_team,
        r.name as race_name,
        r.date as race_date,
        r.circuit as race_circuit
       FROM predictions p
       JOIN drivers d ON p.driver_id = d.id
       JOIN races r ON p.race_id = r.id
       WHERE p.race_id = ?
       ORDER BY p.predicted_position`).bind(raceId).all();
    return new Response(JSON.stringify({ predictions }), {
      headers: { "Content-Type": "application/json" }
    });
  } catch (error) {
    return new Response(JSON.stringify({ error: "Failed to fetch latest predictions" }), {
      status: 500,
      headers: { "Content-Type": "application/json" }
    });
  }
}
__name(handleLatestPredictions, "handleLatestPredictions");

// packages/worker/dist/handlers/results.js
async function handleResults(request, env) {
  const url = new URL(request.url);
  const raceId = url.pathname.split("/").pop();
  try {
    const { results } = await env.DB.prepare(`SELECT 
        rr.*,
        d.name as driver_name,
        d.code as driver_code,
        d.team as driver_team,
        r.name as race_name,
        r.date as race_date
       FROM race_results rr
       JOIN drivers d ON rr.driver_id = d.id
       JOIN races r ON rr.race_id = r.id
       WHERE rr.race_id = ?
       ORDER BY rr.position`).bind(raceId).all();
    return new Response(JSON.stringify({ results }), {
      headers: { "Content-Type": "application/json" }
    });
  } catch (error) {
    return new Response(JSON.stringify({ error: "Failed to fetch results" }), {
      status: 500,
      headers: { "Content-Type": "application/json" }
    });
  }
}
__name(handleResults, "handleResults");

// packages/worker/dist/handlers/drivers.js
async function handleDrivers(request, env) {
  try {
    const { results } = await env.DB.prepare(`SELECT 
        d.*,
        COUNT(DISTINCT p.race_id) as total_predictions,
        AVG(p.predicted_position) as avg_predicted_position,
        COUNT(DISTINCT rr.race_id) as total_races,
        AVG(rr.position) as avg_actual_position,
        SUM(rr.points) as total_points
       FROM drivers d
       LEFT JOIN predictions p ON d.id = p.driver_id
       LEFT JOIN race_results rr ON d.id = rr.driver_id
       GROUP BY d.id
       ORDER BY total_points DESC`).all();
    return new Response(JSON.stringify({ drivers: results }), {
      headers: { "Content-Type": "application/json" }
    });
  } catch (error) {
    return new Response(JSON.stringify({ error: "Failed to fetch drivers" }), {
      status: 500,
      headers: { "Content-Type": "application/json" }
    });
  }
}
__name(handleDrivers, "handleDrivers");

// packages/worker/dist/handlers/races.js
async function handleRaces(request, env) {
  try {
    const url = new URL(request.url);
    const season = url.searchParams.get("season") || (/* @__PURE__ */ new Date()).getFullYear().toString();
    const { results } = await env.DB.prepare(`SELECT 
        r.*,
        COUNT(DISTINCT p.id) as prediction_count,
        COUNT(DISTINCT rr.id) as result_count,
        CASE 
          WHEN r.date < date('now') THEN 'completed'
          WHEN r.date = date('now') THEN 'today'
          ELSE 'upcoming'
        END as status
       FROM races r
       LEFT JOIN predictions p ON r.id = p.race_id
       LEFT JOIN race_results rr ON r.id = rr.race_id
       WHERE r.season = ?
       GROUP BY r.id
       ORDER BY r.round`).bind(season).all();
    return new Response(JSON.stringify({ races: results }), {
      headers: { "Content-Type": "application/json" }
    });
  } catch (error) {
    return new Response(JSON.stringify({ error: "Failed to fetch races" }), {
      status: 500,
      headers: { "Content-Type": "application/json" }
    });
  }
}
__name(handleRaces, "handleRaces");

// packages/worker/dist/handlers/analytics.js
async function handleAccuracy(request, env) {
  try {
    const { results: metrics } = await env.DB.prepare(`SELECT 
        mm.*,
        r.name as race_name,
        r.date as race_date
       FROM model_metrics mm
       LEFT JOIN races r ON mm.race_id = r.id
       ORDER BY mm.created_at DESC
       LIMIT 20`).all();
    const { results: accuracy } = await env.DB.prepare(`SELECT 
        r.name as race_name,
        r.date as race_date,
        AVG(ABS(p.predicted_position - rr.position)) as avg_position_error,
        AVG(ABS(p.predicted_time - rr.time)) as avg_time_error,
        COUNT(*) as prediction_count
       FROM predictions p
       JOIN race_results rr ON p.race_id = rr.race_id AND p.driver_id = rr.driver_id
       JOIN races r ON p.race_id = r.id
       GROUP BY p.race_id
       ORDER BY r.date DESC
       LIMIT 10`).all();
    return new Response(JSON.stringify({
      model_metrics: metrics,
      prediction_accuracy: accuracy
    }), {
      headers: { "Content-Type": "application/json" }
    });
  } catch (error) {
    return new Response(JSON.stringify({ error: "Failed to fetch analytics" }), {
      status: 500,
      headers: { "Content-Type": "application/json" }
    });
  }
}
__name(handleAccuracy, "handleAccuracy");

// packages/worker/dist/index.js
var router = e();
var corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type"
};
router.options("*", () => {
  return new Response(null, { headers: corsHeaders });
});
router.get("/api/predictions/latest", handleLatestPredictions);
router.get("/api/predictions/:raceId", handlePredictions);
router.get("/api/results/:raceId", handleResults);
router.get("/api/drivers", handleDrivers);
router.get("/api/races", handleRaces);
router.get("/api/analytics/accuracy", handleAccuracy);
router.get("/api/health", () => {
  return new Response(JSON.stringify({ status: "ok" }), {
    headers: { ...corsHeaders, "Content-Type": "application/json" }
  });
});
router.all("*", () => {
  return new Response("Not Found", { status: 404 });
});
var dist_default = {
  async fetch(request, env) {
    return router.handle(request, env).then((response) => {
      Object.entries(corsHeaders).forEach(([key, value]) => {
        response.headers.set(key, value);
      });
      return response;
    });
  }
};

// packages/worker/node_modules/wrangler/templates/middleware/middleware-ensure-req-body-drained.ts
var drainBody = /* @__PURE__ */ __name(async (request, env, _ctx, middlewareCtx) => {
  try {
    return await middlewareCtx.next(request, env);
  } finally {
    try {
      if (request.body !== null && !request.bodyUsed) {
        const reader = request.body.getReader();
        while (!(await reader.read()).done) {
        }
      }
    } catch (e2) {
      console.error("Failed to drain the unused request body.", e2);
    }
  }
}, "drainBody");
var middleware_ensure_req_body_drained_default = drainBody;

// packages/worker/node_modules/wrangler/templates/middleware/middleware-miniflare3-json-error.ts
function reduceError(e2) {
  return {
    name: e2?.name,
    message: e2?.message ?? String(e2),
    stack: e2?.stack,
    cause: e2?.cause === void 0 ? void 0 : reduceError(e2.cause)
  };
}
__name(reduceError, "reduceError");
var jsonError = /* @__PURE__ */ __name(async (request, env, _ctx, middlewareCtx) => {
  try {
    return await middlewareCtx.next(request, env);
  } catch (e2) {
    const error = reduceError(e2);
    return Response.json(error, {
      status: 500,
      headers: { "MF-Experimental-Error-Stack": "true" }
    });
  }
}, "jsonError");
var middleware_miniflare3_json_error_default = jsonError;

// .wrangler/tmp/bundle-OzcOnk/middleware-insertion-facade.js
var __INTERNAL_WRANGLER_MIDDLEWARE__ = [
  middleware_ensure_req_body_drained_default,
  middleware_miniflare3_json_error_default
];
var middleware_insertion_facade_default = dist_default;

// packages/worker/node_modules/wrangler/templates/middleware/common.ts
var __facade_middleware__ = [];
function __facade_register__(...args) {
  __facade_middleware__.push(...args.flat());
}
__name(__facade_register__, "__facade_register__");
function __facade_invokeChain__(request, env, ctx, dispatch, middlewareChain) {
  const [head, ...tail] = middlewareChain;
  const middlewareCtx = {
    dispatch,
    next(newRequest, newEnv) {
      return __facade_invokeChain__(newRequest, newEnv, ctx, dispatch, tail);
    }
  };
  return head(request, env, ctx, middlewareCtx);
}
__name(__facade_invokeChain__, "__facade_invokeChain__");
function __facade_invoke__(request, env, ctx, dispatch, finalMiddleware) {
  return __facade_invokeChain__(request, env, ctx, dispatch, [
    ...__facade_middleware__,
    finalMiddleware
  ]);
}
__name(__facade_invoke__, "__facade_invoke__");

// .wrangler/tmp/bundle-OzcOnk/middleware-loader.entry.ts
var __Facade_ScheduledController__ = class {
  constructor(scheduledTime, cron, noRetry) {
    this.scheduledTime = scheduledTime;
    this.cron = cron;
    this.#noRetry = noRetry;
  }
  #noRetry;
  noRetry() {
    if (!(this instanceof __Facade_ScheduledController__)) {
      throw new TypeError("Illegal invocation");
    }
    this.#noRetry();
  }
};
__name(__Facade_ScheduledController__, "__Facade_ScheduledController__");
function wrapExportedHandler(worker) {
  if (__INTERNAL_WRANGLER_MIDDLEWARE__ === void 0 || __INTERNAL_WRANGLER_MIDDLEWARE__.length === 0) {
    return worker;
  }
  for (const middleware of __INTERNAL_WRANGLER_MIDDLEWARE__) {
    __facade_register__(middleware);
  }
  const fetchDispatcher = /* @__PURE__ */ __name(function(request, env, ctx) {
    if (worker.fetch === void 0) {
      throw new Error("Handler does not export a fetch() function.");
    }
    return worker.fetch(request, env, ctx);
  }, "fetchDispatcher");
  return {
    ...worker,
    fetch(request, env, ctx) {
      const dispatcher = /* @__PURE__ */ __name(function(type, init) {
        if (type === "scheduled" && worker.scheduled !== void 0) {
          const controller = new __Facade_ScheduledController__(
            Date.now(),
            init.cron ?? "",
            () => {
            }
          );
          return worker.scheduled(controller, env, ctx);
        }
      }, "dispatcher");
      return __facade_invoke__(request, env, ctx, dispatcher, fetchDispatcher);
    }
  };
}
__name(wrapExportedHandler, "wrapExportedHandler");
function wrapWorkerEntrypoint(klass) {
  if (__INTERNAL_WRANGLER_MIDDLEWARE__ === void 0 || __INTERNAL_WRANGLER_MIDDLEWARE__.length === 0) {
    return klass;
  }
  for (const middleware of __INTERNAL_WRANGLER_MIDDLEWARE__) {
    __facade_register__(middleware);
  }
  return class extends klass {
    #fetchDispatcher = (request, env, ctx) => {
      this.env = env;
      this.ctx = ctx;
      if (super.fetch === void 0) {
        throw new Error("Entrypoint class does not define a fetch() function.");
      }
      return super.fetch(request);
    };
    #dispatcher = (type, init) => {
      if (type === "scheduled" && super.scheduled !== void 0) {
        const controller = new __Facade_ScheduledController__(
          Date.now(),
          init.cron ?? "",
          () => {
          }
        );
        return super.scheduled(controller);
      }
    };
    fetch(request) {
      return __facade_invoke__(
        request,
        this.env,
        this.ctx,
        this.#dispatcher,
        this.#fetchDispatcher
      );
    }
  };
}
__name(wrapWorkerEntrypoint, "wrapWorkerEntrypoint");
var WRAPPED_ENTRY;
if (typeof middleware_insertion_facade_default === "object") {
  WRAPPED_ENTRY = wrapExportedHandler(middleware_insertion_facade_default);
} else if (typeof middleware_insertion_facade_default === "function") {
  WRAPPED_ENTRY = wrapWorkerEntrypoint(middleware_insertion_facade_default);
}
var middleware_loader_entry_default = WRAPPED_ENTRY;
export {
  __INTERNAL_WRANGLER_MIDDLEWARE__,
  middleware_loader_entry_default as default
};
//# sourceMappingURL=index.js.map
