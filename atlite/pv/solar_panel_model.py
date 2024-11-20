# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT

import numpy as np

# Huld model was copied from gsee -- global solar energy estimator
# by Stefan Pfenninger
# https://github.com/renewables-ninja/gsee/blob/master/gsee/pv.py


def _power_huld(irradiance, t_amb, pc):
    """
    AC power per capacity predicted by Huld model, based on W/m2 irradiance.

    Maximum power point tracking is assumed.

    [1] Huld, T. et al., 2010. Mapping the performance of PV modules,
    effects of module type and data averaging. Solar Energy, 84(2),
    p.324-338. DOI: 10.1016/j.solener.2009.12.002
    """
    # normalized module temperature
    T_ = (pc["c_temp_amb"] * t_amb + pc["c_temp_irrad"] * irradiance) - pc["r_tmod"]

    # normalized irradiance
    G_ = irradiance / pc["r_irradiance"]

    log_G_ = np.log(G_.where(G_ > 0))
    # NB: np.log without base implies base e or ln
    eff = (
        1
        + pc["k_1"] * log_G_
        + pc["k_2"] * (log_G_) ** 2
        + T_ * (pc["k_3"] + pc["k_4"] * log_G_ + pc["k_5"] * log_G_**2)
        + pc["k_6"] * (T_**2)
    )

    eff = eff.fillna(0.0).clip(min=0)

    da = G_ * eff * pc.get("inverter_efficiency", 1.0)
    da.attrs["units"] = "kWh/kWp"
    da = da.rename("specific generation")

    return da


def _power_bofinger(irradiance, t_amb, pc):
    """
    AC power per capacity predicted by bofinger model, based on W/m2
    irradiance.

    Maximum power point tracking is assumed.

    [2] Hans Beyer, Gerd Heilscher and Stefan Bofinger, 2004. A robust
    model for the MPP performance of different types of PV-modules
    applied for the performance check of grid connected systems.
    """
    fraction = (pc["NOCT"] - pc["Tamb"]) / pc["Intc"]

    eta_ref = (
        pc["A"]
        + pc["B"] * irradiance
        + pc["C"] * np.log(irradiance.where(irradiance != 0))
    )
    eta = (
        eta_ref
        * (1.0 + pc["D"] * (fraction * irradiance + (t_amb - pc["Tstd"])))
        / (1.0 + pc["D"] * fraction / pc["ta"] * eta_ref * irradiance)
    ).fillna(0)

    capacity = (pc["A"] + pc["B"] * 1000.0 + pc["C"] * np.log(1000.0)) * 1e3
    power = irradiance * eta * (pc.get("inverter_efficiency", 1.0) / capacity)
    power = power.where(irradiance >= pc["threshold"], 0)
    return power.rename("AC power")


def SolarPanelModel(ds, irradiance, pc):
    model = pc.get("model", "huld")

    if model == "huld":
        return _power_huld(irradiance, ds["temperature"], pc)
    elif model == "bofinger":
        return _power_bofinger(irradiance, ds["temperature"], pc)
    else:
        AssertionError(f"Unknown panel model: {model}")
