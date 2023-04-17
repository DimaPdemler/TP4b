from numpy import sqrt, cos, pi

from vector import array

def deltaR(*args):
    """
    Arguments:
        -eta1 : pseudorapidity of the first particle
        -eta2 : pseudorapidity of the second particle
        -phi1 : azimuthal angle of the first particle
        -phi2 : azimuthal angle of the second particle
    """
    if len(args) == 1:
        if len(args[0]) != 4:
            raise TypeError("Wrong number of arguments")
        eta1 = args[0][0]
        eta2 = args[0][1]
        phi1 = args[0][2]
        phi2 = args[0][3]
    elif len(args) == 4:
        eta1 = args[0]
        eta2 = args[1]
        phi1 = args[2]
        phi2 = args[3]
    else:
        raise TypeError("Wrong number of arguments")
    return sqrt((eta1-eta2)**2+((((phi1-phi2)+ pi) % (2*pi)) - pi)**2) # type: ignore

def sum_pt(*args):
    """
    Aguments : 
        -pts : transverse momentum of the particles
        -phis : azimuthal angles of the particles
        -etas : pseudorapidity of the particles 
        -masses : masses of the particles 
    All arguments have 2 coordinates :
        -the first component corresponds to the type of particle (muon, tau, MET)
        -The second coordinate corresponds to the event.
    """
    if len(args) == 1:
        if len(args[0]) != 4:
            raise TypeError("Wrong number of arguments")
        pts = args[0][0]
        phis = args[0][1]
        etas = args[0][2]
        masses = args[0][3]
    elif len(args) == 4:
        pts = args[0]
        phis = args[1]
        etas = args[2]
        masses = args[3]
    else:
        raise TypeError("Wrong number of arguments")
    p_tot = array({"pt" : pts[0],
                          "phi" : phis[0],
                          "eta" : etas[0],
                          "M" : masses[0]})
    for i in range(1,len(masses)):
        p_tot += array({"pt" : pts[i],
                               "phi" : phis[i],
                               "eta" : etas[i],
                               "M" : masses[i]}) # type: ignore
    return p_tot.pt # type: ignore

def transverse_mass(*args):
    """
    Arguments : 
        -pt_1 : transverse momentum of the first particle
        -pt_2 : transverse momentum of the second particle
        -phi_1 : azimuthal angle of the first particle
        -phi_2 : azimuthal angle of the second particle
    """
    if len(args) == 1:
        if len(args[0]) != 4:
            raise TypeError("Wrong number of arguments")
        pt_1 = args[0][0]
        pt_2 = args[0][1]
        phi_1 = args[0][2]
        phi_2 = args[0][3]
    elif len(args) == 4:
        pt_1 = args[0]
        pt_2 = args[1]
        phi_1 = args[2]
        phi_2 = args[3]
    else:
        raise TypeError("Wrong number of arguments")
    return sqrt(2.0*pt_1*pt_2*(1.0 - cos(phi_1-phi_2)))

def total_transverse_mass(*args):
    """
    Arguments : 
        -pt_1 : transverse momentum of the first particle 
        -pt_2 : transverse momentum of the second particle
        -pt_3 : transverse momentum of the third particle
        -pt_miss : missing transverse momentum 
        -phi_1 : azimuthal angle of the first particle
        -phi_2 : azimuthal angle of the second particle
        -phi_3 : azimuthal angle of the third particle
        -phi_miss : azimuthal angle of missing particles
    """
    if len(args) == 1:
        if len(args[0]) != 8:
            raise TypeError("Wrong number of arguments")
        pt_1 = args[0][0]
        pt_2 = args[0][1]
        pt_3 = args[0][2]
        pt_miss = args[0][3]
        phi_1 = args[0][4]
        phi_2 = args[0][5]
        phi_3 = args[0][6]
        phi_miss = args[0][7]
    elif len(args) == 8:
        pt_1 = args[0]
        pt_2 = args[1]
        pt_3 = args[2]
        pt_miss = args[3]
        phi_1 = args[4]
        phi_2 = args[5]
        phi_3 = args[6]
        phi_miss = args[7]
    else:
        raise TypeError("Wrong number of arguments")
    return sqrt(transverse_mass(pt_1, pt_2, phi_1, phi_2)**2 + transverse_mass(pt_1, pt_3, phi_1, phi_3)**2 + 
                transverse_mass(pt_2, pt_3, phi_2, phi_3)**2 + transverse_mass(pt_1, pt_miss, phi_1, phi_miss)**2 +
                transverse_mass(pt_2, pt_miss, phi_2, phi_miss)**2 + transverse_mass(pt_3, pt_miss, phi_3, phi_miss)**2)