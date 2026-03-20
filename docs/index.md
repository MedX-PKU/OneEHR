<div class="landing-hero">
  <div class="landing-hero-copy">
    <p class="landing-eyebrow">EHR AI platform</p>
    <h1>OneEHR</h1>
    <p class="landing-lede">
      From standardized EHR tables to reproducible runs, structured analysis, and cross-system
      comparison across ML/DL models and LLM systems.
    </p>
    <p class="landing-body">
      OneEHR is a Python platform for longitudinal EHR experiments. It provides shared
      infrastructure for preprocessing, modeling, testing, and analysis on one shared run contract so
      the CLI and notebooks all read the same saved artifacts.
    </p>
    <div class="landing-actions">
      <a class="landing-button landing-button-primary" href="./getting-started/quickstart/">Run the quickstart</a>
      <a class="landing-button landing-button-secondary" href="./guide/core-workflows/">Core workflows guide</a>
    </div>
  </div>
  <div class="landing-hero-panel">
    <div class="landing-badge-row">
      <span class="landing-badge">Python 3.12</span>
      <span class="landing-badge">TOML config</span>
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
        <span class="landing-stat-meta">Parquet + JSON</span>
      </article>
      <article class="landing-stat-card">
        <span class="landing-stat-label">Models</span>
        <strong>6 built-in</strong>
        <span class="landing-stat-meta">xgboost, catboost, gru, lstm, tcn, transformer</span>
      </article>
      <article class="landing-stat-card">
        <span class="landing-stat-label">System layer</span>
        <strong>Cross-system comparison</strong>
        <span class="landing-stat-meta">Same samples, same scoring contract</span>
      </article>
    </div>
  </div>
</div>

## Why OneEHR

Most EHR projects do not fail because a model cannot be trained. They fail because preprocessing, splits, and analysis all drift into different formats owned by different scripts. OneEHR keeps those stages on one shared run contract so that a run remains reproducible and inspectable long after training finishes.

<div class="feature-grid">
  <article class="feature-card">
    <p class="feature-kicker">Standardize first</p>
    <h3>Event-table in, not dataset magic</h3>
    <p>Prepare normalized EHR tables once, then reuse the same inputs across preprocess, training, testing, and analysis.</p>
  </article>
  <article class="feature-card">
    <p class="feature-kicker">Shared contract</p>
    <h3>One shared run contract across every interface</h3>
    <p>The CLI and notebooks all read the same run directory instead of parallel export formats.</p>
  </article>
  <article class="feature-card">
    <p class="feature-kicker">Comparable outputs</p>
    <h3>Unified predictions and structured analysis</h3>
    <p>A single predictions.parquet with a system column enables cross-system comparison. Analysis modules produce JSON artifacts that stay explorable after the experiment is over.</p>
  </article>
  <article class="feature-card">
    <p class="feature-kicker">Cross-system comparison</p>
    <h3>Unified scoring across systems</h3>
    <p>ML/DL models and LLM systems are tested on the same split with the same metrics, so comparisons stay fair and reproducible.</p>
  </article>
</div>

## Workflow At A Glance

<div class="workflow-grid">
  <article class="workflow-step">
    <span class="workflow-step-no">01</span>
    <h3>Preprocess</h3>
    <p>Materialize binned features and labels from standardized EHR tables.</p>
  </article>
  <article class="workflow-step">
    <span class="workflow-step-no">02</span>
    <h3>Train</h3>
    <p>Fit tabular and deep learning models from a TOML experiment contract.</p>
  </article>
  <article class="workflow-step">
    <span class="workflow-step-no">03</span>
    <h3>Test</h3>
    <p>Evaluate all trained models and configured systems on the held-out test split.</p>
  </article>
  <article class="workflow-step">
    <span class="workflow-step-no">04</span>
    <h3>Analyze</h3>
    <p>Write structured analysis outputs for cross-system comparison and feature importance.</p>
  </article>
</div>

## Choose Your Entry Point

<div class="entry-grid">
  <article class="entry-card">
    <h3><a href="./getting-started/quickstart/">Quickstart</a></h3>
    <p>Use the bundled TJH example config for the shortest path from raw tables to a complete run directory.</p>
  </article>
  <article class="entry-card">
    <h3><a href="./guide/core-workflows/">Core Workflows</a></h3>
    <p>Understand the standard preprocess, train, test, and analyze path in detail.</p>
  </article>
  <article class="entry-card">
    <h3><a href="./reference/configuration/">Configuration Reference</a></h3>
    <p>Full TOML option tables for dataset, preprocessing, split, models, trainer, systems, and output.</p>
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
    <p>Saved artifacts are machine-readable (Parquet + JSON), so downstream automation does not depend on hidden cells.</p>
  </article>
  <article class="principle-card">
    <h3>Cross-system comparison is built in</h3>
    <p>ML/DL models and LLM systems produce predictions in the same format, enabling fair comparison via the test and analyze commands.</p>
  </article>
</div>

## Start Here

- Use [Installation](getting-started/installation.md) to set up Python 3.12, `uv`, and optional extras.
- Use [Quickstart](getting-started/quickstart.md) for a runnable end-to-end example.
- Use [Configuration Reference](reference/configuration.md) if you are authoring experiment TOML files.
- Use [Artifacts Reference](reference/artifacts.md) if you need the precise on-disk contract for tooling.
