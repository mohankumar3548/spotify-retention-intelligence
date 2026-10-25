/* Spotify Churn Radar: sample data generator and algorithm for Spotify. */
const TOOL = {
  num: 2, area: "Marketing", product: "Churn & Retention Intelligence", seed: 42,
  title: "Spotify Churn Radar", company: "Spotify",
  why: "Spotify runs on monthly subscriptions and competes with Apple Music and YouTube Music for the same listeners. This tool scores every subscriber for churn risk, explains why, and assigns a retention play.",
  statement: "A subscription business is experiencing increasing customer churn despite continued marketing expenditure. Develop an AI system that identifies behavioral patterns associated with churn, segments at-risk customers, determines potential causes, and generates personalized retention strategies.",
  dataNote: "3,000 Premium subscribers with 90 days of listening, billing and support history, scaled 2,000x to a 6M-subscriber market (sample data).",
  steps: [
    ["Collect subscriber data", "Listening, skips, logins, billing, plan, support tickets, reviews"],
    ["Engineer behaviour features", "Change in weekly listening, days since login, failed payments"],
    ["Train churn classifier", "Logistic regression, 70/30 split, class-weighted"],
    ["Explain each user", "Per-user feature contributions (SHAP-style) and global drivers"],
    ["Time to churn", "Monthly hazard turned into expected months until cancel"],
    ["Segment at-risk users", "Group by each user's top churn driver"],
    ["Text signals", "Keyword sentiment on reviews and tickets"],
    ["Retention plays", "Rules + AI analyst pick an action per segment"]
  ],
  params: [
    { id: "th", label: "At-risk threshold", min: 0.3, max: 0.8, step: 0.05, value: 0.35, fmt: v => F.pct(v, 0) },
    { id: "save", label: "Expected save rate of plays", min: 10, max: 50, step: 5, value: 25, fmt: v => v + "%" }
  ],
  questions: ["What is the biggest reason people cancel?", "Which segment should we target first and why?", "Write a win-back email for the price-sensitive group", "How accurate is the churn model?"],
  generate(r) {
    const plans = ["Individual", "Duo", "Family", "Student"], prices = { Individual: 11.99, Duo: 16.99, Family: 19.99, Student: 5.99 };
    const neg = ["too expensive", "price went up", "ads came back", "app keeps crashing", "cannot find new music", "same songs every day", "payment failed again", "support never replied"], pos = ["love my daily mix", "great podcasts", "worth the price", "discover weekly is perfect"];
    const users = [];
    for (let i = 0; i < 3000; i++) {
      const plan = r.wpick(plans, [5, 1.5, 2.5, 1.5]); const tenure = Math.max(1, Math.round(r.u(1, 60)));
      const hrs = Math.max(0.3, r.n(14, 6)); const chg = S.clamp(r.n(-0.05, 0.3), -0.9, 1.2);
      const skip = S.clamp(r.n(0.25, 0.1), 0.02, 0.8); const idle = Math.max(0, Math.round(Math.abs(r.n(0, 6)) + (chg < -0.4 ? r.u(4, 20) : 0)));
      const failPay = r.chance(0.08) ? r.int(1, 3) : 0; const tickets = r.chance(0.15) ? r.int(1, 4) : 0; const priceHit = plan !== "Student" && r.chance(0.35) ? 1 : 0;
      const newMusic = S.clamp(r.n(0.3, 0.12), 0, 0.9);
      const z = -2.6 - 2.2 * chg + 3.2 * (skip - 0.25) + 0.11 * idle + 0.9 * failPay + 0.38 * tickets + 1.1 * priceHit - 0.018 * tenure - 0.045 * hrs - 2.2 * (newMusic - 0.3) + (plan === "Family" ? -0.5 : 0);
      const churn = r() < S.sigmoid(z) ? 1 : 0;
      const sentiment = r() < S.sigmoid(z + 1) ? r.pick(neg) : r.pick(pos);
      users.push({ id: "U" + (10000 + i), plan, price: prices[plan], tenure, hrs, chg, skip, idle, failPay, tickets, priceHit, newMusic, churn, review: sentiment });
    }
    return { users };
  },
  analyze(D, P, r) {
    const U = D.users, names = ["Weekly listening (hrs)", "Listening change (90d)", "Skip rate", "Days since last login", "Failed payments", "Support tickets", "Hit by price increase", "Tenure (months)", "New-music share", "Family plan"];
    const x = u => [u.hrs, u.chg, u.skip, u.idle, u.failPay, u.tickets, u.priceHit, u.tenure, u.newMusic, u.plan === "Family" ? 1 : 0];
    const idx = S.range(U.length); for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1));[idx[i], idx[j]] = [idx[j], idx[i]]; }
    const tr = idx.slice(0, 2100), te = idx.slice(2100);
    const m = M.logit(tr.map(i => x(U[i])), tr.map(i => U[i].churn), { iters: 350, lr: 0.5 });
    const ps = U.map(u => m.predict(x(u)));
    const auc = M.auc(te.map(i => ps[i]), te.map(i => U[i].churn));
    const base = S.mean(U.map(u => u.churn));
    const glob = names.map((nm, j) => ({ name: nm, w: m.w[j], imp: S.mean(U.map(u => Math.abs(m.contrib(x(u))[j]))) })).sort((a, b) => b.imp - a.imp);
    // segments by top positive driver
    const segOf = { 1: "Disengaged listeners", 3: "Disengaged listeners", 0: "Disengaged listeners", 2: "Content mismatch", 8: "Content mismatch", 4: "Payment friction", 5: "Service issues", 6: "Price-sensitive", 7: "New and unsettled", 9: "Price-sensitive" };
    const play = { "Disengaged listeners": "Personal re-engagement: a fresh Daily Mix and a push for the artist they played most", "Content mismatch": "Discovery boost: tuned Discover Weekly and a 'new for you' playlist", "Payment friction": "Payment rescue: retry on payday, card update reminder, 7-day grace period", "Service issues": "Support recovery: a priority callback and a free month for unresolved tickets", "Price-sensitive": "Plan-switch offer: a Duo or annual plan at a lower monthly cost before renewal", "New and unsettled": "Onboarding nudge: build their first playlists in the first 30 days" };
    const atRisk = U.map((u, i) => ({ u, p: ps[i], c: m.contrib(x(u)) })).filter(o => o.p >= P.th && !o.u.churn);
    atRisk.forEach(o => { let bj = 0, bv = -Infinity; o.c.forEach((v, j) => { if (v > bv) { bv = v; bj = j; } }); o.seg = segOf[bj]; o.driver = names[bj]; o.months = Math.max(0.5, Math.log(0.5) / Math.log(1 - Math.min(0.95, o.p * 0.6))); });
    const segs = [...S.groupBy(atRisk, o => o.seg)].map(([k, v]) => ({ k, n: v.length, mrr: S.sum(v.map(o => o.u.price)), p: S.mean(v.map(o => o.p)), months: S.mean(v.map(o => o.months)) })).sort((a, b) => b.mrr - a.mrr);
    const SC = 2000; segs.forEach(s => { s.n *= SC; s.mrr *= SC; });
    const mrrRisk = S.sum(atRisk.map(o => o.u.price)) * SC, saved = mrrRisk * P.save / 100 * 12;
    // text keyword sentiment
    const kw = { "too expensive": "Price", "price went up": "Price", "ads came back": "Price", "app keeps crashing": "App quality", "cannot find new music": "Content", "same songs every day": "Content", "payment failed again": "Billing", "support never replied": "Support" };
    const churned = U.filter(u => u.churn); const themes = {}; churned.forEach(u => { const t = kw[u.review]; if (t) themes[t] = (themes[t] || 0) + 1; });
    const themeArr = Object.entries(themes).map(([k, v]) => ({ label: k, value: v / churned.length })).sort((a, b) => b.value - a.value);
    // histogram
    const bins = S.range(10).map(b => ({ label: (b * 10) + "%", value: ps.filter(p => p >= b / 10 && p < (b + 1) / 10).length, tone: b / 10 >= P.th - 1e-9 ? "bad" : "mut" }));
    const top = [...atRisk].sort((a, b) => b.p * b.u.price - a.p * a.u.price).slice(0, 10);
    return {
      steps: [`${U.length} subscribers, ${F.pct(base)} churned in 90 days`, "10 features built per subscriber", `Test AUC ${auc.toFixed(3)} on ${te.length} held-out users`, `Top driver: ${glob[0].name}`, `Median expected time to cancel ${S.quantile(atRisk.map(o => o.months), 0.5).toFixed(1)} months`, `${segs.length} segments, ${atRisk.length} of the sample at risk`, `Top complaint theme among churners: ${themeArr[0]?.label}`, `${segs.length} plays assigned`],
      kpis: [
        { label: "90-day churn", value: F.pct(base), note: `${churned.length} of ${U.length} subscribers`, tone: "bad" },
        { label: "Active users at risk", value: F.num(atRisk.length * SC), note: `Score ≥ ${F.pct(P.th, 0)}, scaled to a 6M base` , tone: "warn"},
        { label: "Revenue at risk", value: F.money(mrrRisk * 12), note: "Annualised subscription value" },
        { label: "Revenue saved", value: F.money(saved), note: `At ${P.save}% play success`, tone: "good" },
        { label: "Model AUC", value: auc.toFixed(2), note: "1.0 = perfect, 0.5 = coin flip" }
      ],
      sections: [
        { pair: [
          { title: "What drives churn", note: "Average size of each feature's push on a user's score. Red raises risk, black lowers it.", chart: C.bar(glob.map(g => ({ label: g.name, value: g.imp, tone: g.w > 0 ? "bad" : "" })), { fmt: v => v.toFixed(2) }) },
          { title: "Risk score distribution", note: "All 3,000 subscribers. Red bars sit above the at-risk threshold.", chart: C.vbar(bins, { fmt: v => F.num(v) }) }
        ] },
        { pair: [
          { title: "At-risk segments", note: "Active subscribers grouped by their single biggest churn driver.", chart: C.bar(segs.map(s => ({ label: s.k, value: s.n, text: `${s.n} users · ${F.money(s.mrr * 12)}/yr` })), { fmt: v => F.num(v), labelW: 160 }) },
          { title: "What churners said", note: "Share of churned users whose last review or ticket hit each theme.", chart: C.bar(themeArr.map(t => ({ label: t.label, value: t.value, tone: "c3" })), { fmt: v => F.pct(v, 0) }) }
        ] },
        { title: "Retention plan by segment", html: T.table(["Segment", "Users", "Avg risk", "Months to cancel", "Revenue at risk", "Play"], segs.map(s => [s.k, F.num(s.n), F.pct(s.p, 0), s.months.toFixed(1), F.money(s.mrr * 12), play[s.k]]), { right: [1, 2, 3, 4] }) },
        { title: "Highest-value users to save first", note: "Ranked by churn probability × monthly price.", html: T.table(["User", "Plan", "Risk", "Top driver", "Action"], top.map(o => [o.u.id, o.u.plan, T.pill(F.pct(o.p, 0), o.p > 0.7 ? "bad" : "warn"), o.driver, play[o.seg].split(":")[0]]), { right: [] }) }
      ],
      recs: segs.slice(0, 4).map(s => ({ title: `${s.k}: ${play[s.k].split(":")[0]}`, body: `${s.n} users at ${F.pct(s.p, 0)} average risk, about ${s.months.toFixed(1)} months from cancelling. ${play[s.k].split(": ")[1] || play[s.k]}.`, impact: F.money(s.mrr * 12 * P.save / 100) + "/yr saved", tone: "good" })).concat([{ title: "Stop paying to win back users the model calls safe", body: `${F.num(U.filter((u, i) => ps[i] < 0.15 && !u.churn).length)} users score under 15%. Move them out of discount campaigns and keep the budget for the segments above.`, impact: "Lower CAC", tone: "" }]),
      context: { churn_rate: base, auc, threshold: P.th, drivers: glob.map(g => ({ feature: g.name, importance: +g.imp.toFixed(3), direction: g.w > 0 ? "raises churn" : "lowers churn" })), segments: segs.map(s => ({ segment: s.k, users: s.n, avg_risk: +s.p.toFixed(2), months_to_cancel: +s.months.toFixed(1), annual_revenue_at_risk: Math.round(s.mrr * 12), play: play[s.k] })), complaint_themes: themeArr.map(t => ({ theme: t.label, share_of_churners: +t.value.toFixed(2) })), revenue_saved_per_year: Math.round(saved) }
    };
  }
};
