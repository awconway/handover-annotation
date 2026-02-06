import dspy


def configure_dspy(lm: dspy.LM) -> None:
    dspy.configure_cache(
        enable_disk_cache=False,
        enable_memory_cache=False,
    )
    dspy.settings.configure(lm=lm)
