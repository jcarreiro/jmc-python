import numpy as np

earth_mean_radius_meters = 6371000

def haversine(radius, phi_0, phi_1, theta_0, theta_1):
    """Computes the length of the arc between the given points on the surface
    of the sphere, using the haversine formula.

    Params
    ------
    radius    Radius of the sphere.
    phi_0     Latitude of first point, in radians.
    phi_1     Latitude of second point, in radians.
    theta_0   Longitude of first point, in radians.
    theta_1   Longitude of second point, in radians.
    """
    return 2.0 * radius * np.arcsin(
        np.sqrt(
            np.sin((phi_1 - phi_0) / 2.0) ** 2 +
            np.cos(phi_0) * np.cos(phi_1) * np.sin((theta_1 - theta_0) / 2.0) ** 2
        )
    )
