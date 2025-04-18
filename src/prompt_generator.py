def build_prompt(state: dict) -> str:
    """Generate text prompt from market state."""
    tpl = (
        "Task: Stock trading agent.\n"
        "Actions: {Buy, Sell, Hold}\n"
        "Current state:\n"
        f"  • Price: ${state['price']:.2f}\n"
        f"  • Volume: {state['volume']:.0f}\n"
        f"  • RSI: {state['rsi']:.2f}\n"
        f"  • Cash: ${state['cash']:.2f}\n"
        f"  • Shares: {state['shares']}\n"
        "Output JSON {'Action': ?}"
    )
    return tpl
