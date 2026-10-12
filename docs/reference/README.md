# Current system reference

ORA supports Native model API policies and scripted infrastructure checks. Its robot integrations are RoboTwin, LIBERO/LIBERO-Pro and RoboDojo; Synthetic provides simulator-free runtime checks.

- [Runtime lifecycle and authority](runtime-lifecycle.md)
- [Public interfaces](public-interfaces.md)
- [Configuration and evidence](configuration-and-evidence.md)
- [Experiment identity](experiment-identity.md)
- [RoboTwin contact recovery](robotwin-contact-recovery.md)
- [RoboDojo motion qualification](robodojo-motion-qualification.md)

The selected default robot baseline is `native_full_explicit_motion_direct_v4`. It binds the current direct Panda, ALOHA closed-skill and ARX closed-skill compositions with Native context. `ora baselines --explain` shows the actual resolved tools and bindings. New benchmark specifications select the baseline by task/provider/resource inputs; they do not copy an older tool surface.

The built-in provider default is `deepseek_flash_openai_low_16k_v1`. The simulator-free `ora run --env synthetic` demonstration has its own small, explicit provider configuration described in the [quickstart](../quickstart.md).

Public namespaces are `ora.*`, including schemas and `ORA_*` runtime variables. External simulator variables keep their benchmark names. Runs from earlier internal schema namespaces must be read at their recorded source revision; this initial distribution makes no implicit evidence migration.

When changing an API, schema, default, context policy, runtime lifecycle, trust boundary or evidence format, update its page here in the same change. The [code map](../code-map.md) identifies implementation owners.
