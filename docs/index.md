<div class="landing-hero">
  <div class="landing-hero-copy">
    <p class="landing-eyebrow">Artifact-first EHR infrastructure</p>
    <h1>OneEHR</h1>
    <p class="landing-lede">
      From standardized EHR tables to reproducible runs, structured analysis, durable case bundles,
      and LLM-agent-ready review workflows.
    </p>
    <p class="landing-body">
      OneEHR is a Python toolkit for longitudinal EHR predictive modeling and analysis. It keeps the
      full workflow on one run contract so the CLI, notebooks, Web UI, and agent workflows all read
      the same saved artifacts instead of inventing parallel formats.
    </p>
    <div class="landing-actions">
      <a class="landing-button landing-button-primary" href="./getting-started/quickstart/">Run the quickstart</a>
      <a class="landing-button landing-button-secondary" href="./guide/agent-workflows/">Inspect agent workflows</a>
      <a class="landing-button landing-button-secondary" href="./guide/webui/">Open the Web UI guide</a>
    </div>
  </div>
  <div class="landing-hero-panel">
    <div class="landing-badge-row">
      <span class="landing-badge">Python 3.12</span>
      <span class="landing-badge">TOML-first</span>
      <span class="landing-badge">CPU-friendly</span>
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
        <span class="landing-stat-label">Review units</span>
        <strong>Durable case bundles</strong>
        <span class="landing-stat-meta">Evidence, predictions, analysis refs</span>
      </article>
      <article class="landing-stat-card">
        <span class="landing-stat-label">Agent layer</span>
        <strong>Predict + review</strong>
        <span class="landing-stat-meta">Grounded in saved case artifacts</span>
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
    <p>Prepare normalized EHR tables once, then reuse the same inputs across preprocess, train, test, analysis, cases, and agents.</p>
  </article>
  <article class="feature-card">
    <p class="feature-kicker">Workflow-first</p>
    <h3>One run contract across every interface</h3>
    <p>The CLI, notebooks, query endpoints, and Web UI all read the same run directory instead of parallel export formats.</p>
  </article>
  <article class="feature-card">
    <p class="feature-kicker">Reviewable outputs</p>
    <h3>Durable cases and structured analysis</h3>
    <p>Failure cases, patient bundles, module outputs, and comparison artifacts stay explorable after the experiment is over.</p>
  </article>
  <article class="feature-card">
    <p class="feature-kicker">Agent-ready</p>
    <h3>LLM workflows grounded in evidence</h3>
    <p>Agent prediction and review operate on saved case artifacts, so prompts, parsed outputs, failures, and metrics are auditable.</p>
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
    <p>Fit classical ML or DL models from a TOML experiment contract.</p>
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
    <h3>Cases</h3>
    <p>Build durable patient-level evidence bundles for inspection, notebooks, and review workflows.</p>
  </article>
  <article class="workflow-step">
    <span class="workflow-step-no">06</span>
    <h3>Agents</h3>
    <p>Run agent prediction and review over saved cases with OpenAI-compatible backends.</p>
  </article>
  <article class="workflow-step">
    <span class="workflow-step-no">07</span>
    <h3>Web UI</h3>
    <p>Browse runs, dashboards, cases, comparisons, and agent artifacts from one read-only console.</p>
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
    <p>Understand the standard preprocess, train, test, analyze, and case-building path in detail.</p>
  </article>
  <article class="entry-card">
    <h3><a href="./guide/agent-workflows/">Agent Workflows</a></h3>
    <p>Configure prompt templates, materialize instances, run agent prediction, and review outputs against case evidence.</p>
  </article>
  <article class="entry-card">
    <h3><a href="./guide/webui/">Web UI</a></h3>
    <p>Serve the browser console for run discovery, module dashboards, case drill-down, and agent artifact browsing.</p>
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
    <h3>Agents are an extension, not a fork</h3>
    <p>The agent layer reuses the same run directory, cases, and evidence rather than bypassing the core EHR workflow.</p>
  </article>
</div>

## Start Here

- Use [Installation](getting-started/installation.md) to set up Python 3.12, `uv`, and optional extras.
- Use [Quickstart](getting-started/quickstart.md) for a runnable end-to-end example.
- Use [Configuration Reference](reference/configuration.md) if you are authoring experiment TOML files.
- Use [Artifacts Reference](reference/artifacts.md) if you need the precise on-disk contract for tooling or UI work.
