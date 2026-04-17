# Default libtorch path from `gliner2 setup`
libtorch := clean(config_directory() / "gliner2/lib/libtorch")

export LIBTORCH := clean(config_directory() / "gliner2/lib/libtorch")
export GLINER2_BENCH_TCH := "1"

clean:
    cargo clean

build: clean
    cargo build --all-targets

build-release: clean
    cargo build --all-targets --release

run *ARGS:
    cargo run -F tch -- {{ARGS}}

clippy:
    cargo clippy --workspace --all-targets -- -D warnings

fmt:
    cargo fmt

setup:
    cargo run -- setup --non-interactive

# Test all
test *ARGS:
    cargo test --all-features {{ARGS}}

bench: build-release
    harness/run_all.sh
    harness/run_compare_all.sh
    harness/run_multitask.sh
    harness/r un_python.sh
    harness/run_throughput.sh

ci: fmt clippy test
