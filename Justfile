# Default libtorch path from `gliner2 setup`
libtorch := clean(config_directory() / "gliner2/lib/libtorch")

clean:
    cargo clean

build: clean
    LIBTORCH={{libtorch}} cargo build --all-targets

build-release: clean
    LIBTORCH={{libtorch}} cargo build --all-targets --release

run *ARGS:
    LIBTORCH={{libtorch}} cargo run -F tch -- {{ARGS}}

clippy:
    LIBTORCH={{libtorch}} cargo clippy --workspace --all-targets -- -D warnings

fmt:
    cargo fmt

setup:
    cargo run -- setup --non-interactive

# Test all
test *ARGS:
    LIBTORCH={{libtorch}} cargo test --all-features {{ARGS}}

bench:
    export GLINER2_BENCH_TCH=1
    export LIBTORCH={{libtorch}}
    harness/run_all.sh
    harness/run_compare_all.sh
    harness/run_multitask.sh
    harness/r un_python.sh
    harness/run_throughput.sh

ci: fmt clippy test
