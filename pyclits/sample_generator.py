import roessler_system



if __name__ == "__main__":
    kwargs = {}
    kwargs["tStop"] = 50
    sol = roessler_system.roessler_oscillator(**kwargs)
    print(sol)
