def rungekutta(dydx, x0, y0, x, h):
    """
    Runge Kutta of 4th the order
    Parameters:
        dydx: Diferential equation
        x0: Initial x
        y0: Initial y
        x: Value of X
        h: Height
    Returns:
        y: Aproximation of y
    """

    n = (int)((x - x0)/h)

    y = y0
    for i in range(1, n + 1):
        "Apply Runge Kutta Formulas to find next value of y"
        k1 = dydx(x0, y)
        k2 = dydx(x0 + 0.5 * h, y + 0.5 * k1)
        k3 = dydx(x0 + 0.5 * h, y + 0.5 * k2)
        k4 = dydx(x0 + h, y + k3)
        print("k1: {}".format(k1))
        print("k2: {}".format(k2))
        print("k3: {}".format(k3))
        print("k4: {}".format(k4))

        y = y + h * (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
 
        print("y{}: {}".format(i, y))
        x0 = x0 + h
    return y

def euler(dydx, x0, y0, x, h):
    """
    Euler Method for solving diferential equations
    Parameters:
        dydx: Diferential equation
        x0: Initial x
        y0: Initial y
        x: Value of X
        h: Height
    Returns:
        y: Aproximation of y
    """
    y = y0
    while x0 < x:
        y = y + h * dydx(x0, y)
        print(y)
        x0 = x0 + h

    return y