# Default libtorch path from `gliner2 setup` # TODO just file should do this setup automagically
libtorch := clean(config_directory() / "gliner2/lib/libtorch")

# Build everything (candle-only launcher + tch engine)
build:
    cargo build -p gliner2
    LIBTORCH={{libtorch}} cargo build -p gliner2-tch

build-release:
    cargo build -p gliner2 --release
    LIBTORCH={{libtorch}} cargo build -p gliner2-tch --release

# Run with tch backend — rebuilds both binaries if source changed
run-tch *ARGS:
    cargo build -p gliner2
    echo "{{libtorch}}"
    LIBTORCH={{libtorch}} cargo build -p gliner2-tch
    LD_LIBRARY_PATH="{{libtorch}}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ./target/debug/gliner2 --backend tch {{ARGS}}

# Run with candle backend (no extra setup needed)
run *ARGS:
    cargo run -p gliner2 -- {{ARGS}}

# Clippy all
clippy:
    LIBTORCH={{libtorch}} cargo clippy --workspace --all-targets -- -D warnings

# Test all
test *ARGS:
    LIBTORCH={{libtorch}} cargo test --all-features {{ARGS}}
