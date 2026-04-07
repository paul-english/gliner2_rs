use anyhow::{Context, Result};
use tch::Device;

/// Parse a device string for the LibTorch (`tch`) backend.
///
/// Accepted forms: `cpu`, `cuda`, `cuda:N`, `mps`, `vulkan`, `auto` (CUDA device 0 if available, else CPU).
pub fn parse_tch_device(s: &str) -> Result<Device> {
    let s = s.trim();
    if s.is_empty() {
        anyhow::bail!(empty_device_help());
    }
    if s == "cpu" {
        return Ok(Device::Cpu);
    }
    if s == "cuda" {
        return Ok(Device::Cuda(0));
    }
    if let Some(rest) = s.strip_prefix("cuda:") {
        let idx = rest
            .trim()
            .parse::<usize>()
            .with_context(|| format!("invalid cuda device index in {s:?}"))?;
        return Ok(Device::Cuda(idx));
    }
    if s == "mps" {
        return Ok(Device::Mps);
    }
    if s == "vulkan" {
        return Ok(Device::Vulkan);
    }
    if s == "auto" {
        return Ok(Device::cuda_if_available());
    }
    anyhow::bail!("unknown device {s:?}; {}", valid_device_forms());
}

fn empty_device_help() -> String {
    format!("device string is empty; {}", valid_device_forms())
}

fn valid_device_forms() -> &'static str {
    "expected one of: cpu, cuda, cuda:N, mps, vulkan, auto"
}

#[cfg(test)]
mod tests {
    use super::parse_tch_device;
    use tch::Device;

    #[test]
    fn parses_cpu() {
        assert_eq!(parse_tch_device("cpu").unwrap(), Device::Cpu);
        assert_eq!(parse_tch_device("  cpu  ").unwrap(), Device::Cpu);
    }

    #[test]
    fn parses_cuda() {
        assert_eq!(parse_tch_device("cuda").unwrap(), Device::Cuda(0));
        assert_eq!(parse_tch_device("cuda:0").unwrap(), Device::Cuda(0));
        assert_eq!(parse_tch_device("cuda:3").unwrap(), Device::Cuda(3));
    }

    #[test]
    fn parses_mps_vulkan_auto() {
        assert_eq!(parse_tch_device("mps").unwrap(), Device::Mps);
        assert_eq!(parse_tch_device("vulkan").unwrap(), Device::Vulkan);
        let _ = parse_tch_device("auto").unwrap();
    }

    #[test]
    fn rejects_unknown() {
        assert!(parse_tch_device("gpu").is_err());
        assert!(parse_tch_device("").is_err());
        assert!(parse_tch_device("cuda:").is_err());
    }
}
