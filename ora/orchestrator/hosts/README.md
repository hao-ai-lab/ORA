# Execution host profiles

Two generic profiles ship here: `local-cpu` (`ora local`, CPU MuJoCo LIBERO)
and `single-gpu` (`ora setup` / `ora run`, one CUDA GPU). For any other host,
write `<host-id>.yaml` from the template `examples/hosts/example-host.yaml` and
point `ORA_HOSTS_DIR` at its directory; a profile there with the same id wins
over the shipped one.
