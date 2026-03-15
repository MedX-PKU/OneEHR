<div class="landing-hero">
  <div class="landing-hero-copy">
    <p class="landing-eyebrow">TOML-first EHR modeling and evaluation infrastructure</p>
    <h1>OneEHR</h1>
    <p class="landing-lede">
      From standardized EHR tables to reproducible runs, structured analysis, and fair evaluation
      across conventional ML/DL models, single-LLM systems, and multi-agent medical AI systems.
    </p>
    <p class="landing-body">
      OneEHR is a TOML-first Python infrastructure toolkit for longitudinal EHR modeling, analysis,
      and cross-system evaluation. It keeps the full workflow on one shared run contract so the CLI,
      notebooks, automation, and web/API layer all read the same saved artifacts instead of inventing
      parallel formats for evaluation or review.
    </p>
    <div class="landing-actions">
      <a class="landing-button landing-button-primary" href="./getting-started/quickstart/">Run the quickstart</a>
      <a class="landing-button landing-button-secondary" href="./guide/eval-workflows/">Inspect evaluation workflows</a>
      <a class="landing-button landing-button-secondary" href="./guide/webui/">Open the web/API guide</a>
    </div>
  </div>
  <div class="landing-hero-panel">
    <div class="landing-badge-row">
      <span class="landing-badge">Python 3.12</span>
      <span class="landing-badge">TOML-first</span>
      <span class="landing-badge">Reproducible</span>
      <span class="landing-badge">OpenAI-compatible</span>
    </div>
    <div class="landing-stats">
      <article class="landing-stat-card">
        <span class="landing-stat-label">Input contract</span>
        <strong>3-table EHR schema</strong>
        <span class="landing-stat-meta"><code>dynamic.csv</code>, <code>static.csv</code>, <code>label.csv</code></span>
      </article>
      <article class="landing-stat-card">
        <span class="landing-stat-label">Run outputs</span>
        <strong>Structured artifacts</strong>
        <span class="landing-stat-meta">JSON, CSV, JSONL, parquet</span>
      </article>
      <article class="landing-stat-card">
        <span class="landing-stat-label">Eval units</span>
        <strong>Frozen instances</strong>
        <span class="landing-stat-meta">Evidence, outputs, traces, reports</span>
      </article>
      <article class="landing-stat-card">
        <span class="landing-stat-label">System layer</span>
        <strong>Model + framework comparison</strong>
        <span class="landing-stat-meta">Same samples, same scoring contract</span>
      </article>
    </div>
  </div>
</div>

## Why OneEHR

Most EHR projects do not fail because a model cannot be trained. They fail because preprocessing, splits, evaluation, dashboards, and prompting all drift into different formats owned by different scripts. OneEHR keeps those stages on one artifact contract so that a run remains reproducible, queryable, and inspectable long after training finishes.

<div class="feature-grid">
  <article class="feature-card">
    <p class="feature-kicker">Standardize first</p>
    <h3>Event-table in, not dataset magic</h3>
    <p>Prepare normalized EHR tables once, then reuse the same inputs across preprocess, training, testing, analysis, and evaluation.</p>
  </article>
  <article class="feature-card">
    <p class="feature-kicker">Shared contract</p>
    <h3>One run contract across every interface</h3>
    <p>The CLI, notebooks, query endpoints, and Web UI all read the same run directory instead of parallel export formats.</p>
  </article>
  <article class="feature-card">
    <p class="feature-kicker">Comparable outputs</p>
    <h3>Frozen instances and structured analysis</h3>
    <p>Saved evidence bundles, analysis modules, leaderboard tables, and pairwise deltas stay explorable after the experiment is over.</p>
  </article>
  <article class="feature-card">
    <p class="feature-kicker">Agentic AI ready</p>
    <h3>Unified scoring across frameworks</h3>
    <p>Evaluate conventional ML/DL baselines, single-LLM systems, and multi-agent medical AI systems on the same frozen instances with saved traces and metrics.</p>
  </article>
</div>

## Workflow At A Glance

<div class="workflow-grid">
  <article class="workflow-step">
    <span class="workflow-step-no">01</span>
    <h3>Preprocess</h3>
    <p>Materialize binned and tabular views from standardized EHR tables.</p>
  </article>
  <article class="workflow-step">
    <span class="workflow-step-no">02</span>
    <h3>Train</h3>
    <p>Fit conventional ML or DL models from a TOML experiment contract.</p>
  </article>
  <article class="workflow-step">
    <span class="workflow-step-no">03</span>
    <h3>Test</h3>
    <p>Evaluate on saved splits or external datasets without losing schema alignment.</p>
  </article>
  <article class="workflow-step">
    <span class="workflow-step-no">04</span>
    <h3>Analyze</h3>
    <p>Write structured module outputs for profiling, audits, temporal views, and interpretability.</p>
  </article>
  <article class="workflow-step">
    <span class="workflow-step-no">05</span>
    <h3>Eval Build</h3>
    <p>Freeze evaluation instances and evidence bundles from the saved run contract.</p>
  </article>
  <article class="workflow-step">
    <span class="workflow-step-no">06</span>
    <h3>Eval Run</h3>
    <p>Execute ML/DL baselines, single-LLM systems, and multi-agent frameworks over the same instances.</p>
  </article>
  <article class="workflow-step">
    <span class="workflow-step-no">07</span>
    <h3>Eval Report</h3>
    <p>Write leaderboard, split metrics, and paired comparisons as stable artifacts.</p>
  </article>
  <article class="workflow-step">
    <span class="workflow-step-no">08</span>
    <h3>Web UI</h3>
    <p>Browse runs, evaluation summaries, analysis dashboards, and comparison artifacts from one read-only interface.</p>
  </article>
</div>

## Choose Your Entry Point

<div class="entry-grid">
  <article class="entry-card">
    <h3><a href="./getting-started/quickstart/">Quickstart</a></h3>
    <p>Use the bundled example config if you want the shortest path from raw tables to a complete run directory.</p>
  </article>
  <article class="entry-card">
    <h3><a href="./guide/core-workflows/">Core Workflows</a></h3>
    <p>Understand the standard preprocess, train, test, analyze, and artifact-production path in detail.</p>
  </article>
  <article class="entry-card">
    <h3><a href="./guide/eval-workflows/">Evaluation Workflows</a></h3>
    <p>Configure frozen instances, compare trained models with framework systems, and inspect saved traces and paired deltas.</p>
  </article>
  <article class="entry-card">
    <h3><a href="./guide/webui/">Web UI</a></h3>
    <p>Serve the browser/API layer for run discovery, module dashboards, evaluation summaries, and comparison drill-down.</p>
  </article>
</div>

## Design Principles

<div class="principles-grid">
  <article class="principle-card">
    <h3>TOML is the experiment contract</h3>
    <p>Configuration is versionable, reviewable, and explicit. If the TOML changes, the experiment changed.</p>
  </article>
  <article class="principle-card">
    <h3>Patient-level leakage prevention</h3>
    <p>Supported split strategies are patient-group aware so that evaluation defaults to safer behavior.</p>
  </article>
  <article class="principle-card">
    <h3>Structured outputs over notebook state</h3>
    <p>Saved artifacts are machine-readable and UI-readable, so downstream automation does not depend on hidden cells.</p>
  </article>
  <article class="principle-card">
    <h3>Evaluation is first-class</h3>
    <p>Framework systems are scored on the same frozen samples and evidence bundles as model baselines, so comparisons stay fair and reproducible.</p>
  </article>
</div>

## Start Here

- Use [Installation](getting-started/installation.md) to set up Python 3.12, `uv`, and optional extras.
- Use [Quickstart](getting-started/quickstart.md) for a runnable end-to-end example.
- Use [Configuration Reference](reference/configuration.md) if you are authoring experiment TOML files.
- Use [Artifacts Reference](reference/artifacts.md) if you need the precise on-disk contract for tooling or UI work.
