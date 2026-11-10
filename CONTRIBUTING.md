# Contributing to Parse

Thanks for helping. Parse aims to bring the best OCR to everyone, and contributions of any size are welcome: bug reports, hard documents that break the parser, docs, and code.

## Ways to help

- **Report a bad parse.** Open an issue with the page (if you can share it), the model and backend you used, and what went wrong. Hard, public examples are especially valuable.
- **Pick an issue.** Issues labelled `good first issue` are a good place to start.
- **Improve docs.** Setup is the biggest pain for users, so clearer install steps help a lot.

## Making a change

1. Fork the repository and create a branch.
2. Make your change and run it on a real document. Include the command and result in the pull request.
3. Run the tests: `pip install -e ".[dev]"`, then `pytest`. They need no network, GPU or API key.
4. No emoji in code or output: Parse runs on Windows terminals too.
5. Open a pull request that describes what changed and why.

## Sign your commits (DCO)

We use the [Developer Certificate of Origin](https://developercertificate.org/) instead of a contributor agreement. Add a sign-off line to each commit to confirm you wrote the code, or have the right to submit it, under the project's license:

```bash
git commit -s -m "Your message"
```

This adds `Signed-off-by: Your Name <you@example.com>`.

## License

By contributing, you agree that your contributions are licensed under the [Apache License 2.0](LICENSE), as described in section 5 of the license.
