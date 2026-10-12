# Security

Please report credential exposure, sandbox/authority bypass or evidence tampering privately through [GitHub private vulnerability reporting](https://github.com/hao-ai-lab/ora-agent/security/advisories/new). If that feature is unavailable, contact a repository maintainer privately before sharing reproductions containing sensitive information. Do not file a public issue with API keys, private endpoints or raw request archives.

Include the affected commit/version, the boundary crossed, a minimal reproduction and the expected authority. Redact credentials and private observations. ORA's model policy must not directly mutate the simulator, access evaluator-private state or set an official verdict.

The initial supported line is 0.1.x. Simulator SDKs, external services and model accounts have their own security/update policies. `ora doctor` reports setup discovery; a passing report is not a security or physical safety certification.
