# Contribute to HeatherDB

Contributions to HeatherDB are welcome: code, documentation, reproducible bug
reports, examples and benchmarks. Every pull request is reviewed manually by a
maintainer before merging. Make the change easy to understand and verify.

## Prepare a focused change

For a substantial feature, API change or storage-format change, open an
[issue](https://github.com/aphorion/heatherdb/issues) to discuss the approach
before implementing it. For a small, well-defined change, open a pull request
directly. Check existing issues and pull requests for related work.

Keep each pull request focused on one problem. Explain the intended behaviour,
the affected code paths and the reason for the approach. Keep unrelated
formatting, dependency changes and refactors out of the diff.

## AI-assisted contributions

You may use AI tools to write, investigate, test or document a contribution.
You remain responsible for everything you submit. The review standard is the
same whether the work is written with AI assistance or without it.

Before submitting AI-assisted work:

- Read and understand every changed line. Be ready to explain the design and
  answer review questions yourself.
- Verify generated code, commands, API names, references and numerical claims
  against the actual source or reproducible evidence.
- Run the relevant checks yourself. Report the commands and observed results;
  distinguish checks you ran from suggestions or checks you could not run.
- Document the behaviour, assumptions, limitations and trade-offs relevant to
  the change. Keep comments useful; do not add narration of obvious code.
- Briefly state in the pull request how AI assisted the work and what you
  independently verified. Prompt transcripts are not required.

Do not submit unreviewed generated patches, invented test results or citations,
speculative abstractions, or large unrelated rewrites. A long generated
explanation is not a substitute for a precise description and evidence.
Maintainers may request a smaller diff or close a submission that cannot be
reviewed reliably.

## Check your work

Use the Rust toolchain declared by the workspace. For a code change, run:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --release --locked --no-fail-fast
```

Add a regression test that exercises the affected behaviour where appropriate.
Use focused tests while developing; report any full-workspace checks you could
not complete. Passing CI supports review and does not replace manual review.

For documentation-only changes, verify commands, links and examples against
the current API. A prose edit does not require running unrelated engine tests.

For changes to memory behaviour, measure the affected recall, learning or
retention task with stated data, seeds, settings and baselines. For performance
claims, include hardware, source revision, build settings, durability, workload,
raw samples and reproducible commands. Compare both versions under matching
conditions. See the [benchmark reference](../reference/benchmarks.md).

For API or stored-format changes, describe compatibility and update the relevant
reference documentation. The [release contract](../reference/v1-release-contract.md)
defines the supported surface.

## Open a reviewable pull request

Target `main`. Include:

1. **Problem and behaviour:** the concrete use case, expected result and what
   the change does. Link the issue if there is one.
2. **Implementation:** the approach, affected components and meaningful
   trade-offs. Explain why any non-obvious code is necessary.
3. **Validation:** exact commands, observed results and reproducible evidence.
   State what remains untested and any known limitations.
4. **Documentation and compatibility:** updated usage or API documentation,
   or a short explanation of why none is needed.
5. **AI assistance:** if used, its role and the parts you independently
   checked. If none was used, say so.

Scale the detail to the change. A small fix can have a short explanation;
a change to persistence or learning needs enough evidence to assess its effects.
Respond to maintainer questions and keep the description current as the patch
changes. Do not open duplicate or repeated generated submissions to seek a
faster review.

## Credit and licensing

We value contributions beyond code, including careful reviews, documentation
and reproducible reports. Preserve attribution for any reused material and
submit only work you have the right to contribute. Contributions to this
repository are provided under its AGPL-3.0-only license.
