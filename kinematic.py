from numpy import array, zeros_like, zeros, sign, sqrt, cos, sin, arctan, exp, pi

from vector import array as v_array
from vector import arr

"Replace *args by the actual variables, and then change function(args) to function(*args) in the call"

def eta_to_theta(eta):
    return 2*arctan(exp(-eta))

#######################################################

def vec_3D(norm, phi, theta, hep_form=True):
    np_output = array([norm*sin(theta)*cos(phi), norm*sin(theta)*cos(phi), norm*cos(theta)])
    if hep_form:
        output = arr({'x':np_output[0], 'y':np_output[1], 'z':np_output[2]})
        return output
    return array

#######################################################

def deltaphi(phi1, phi2):
    """
    Arguments:
        -phi1 : azimuthal angle of the first particle
        -phi2 : azimuthal angle of the second particle    
    """
    # if len(args) == 1:
    #     if len(args[0]) != 2:
    #         raise TypeError("Wrong number of arguments")
    #     phi1 = args[0][0]
    #     phi2 = args[0][1]
    # elif len(args) == 2:
    #     phi1 = args[0]
    #     phi2 = args[1]
    # else:
    #     raise TypeError("Wrong number of arguments")
    return abs((((phi1-phi2)+ pi) % (2*pi)) - pi)

# #######################################################

def deltaphi3(pt_1, pt_2, pt_3, pt_MET,
                phi_1, phi_2, phi_3, phi_MET,
                eta_1, eta_2, eta_3,
                mass_1, mass_2, mass_3):
    """
    Arguments:
        -pt_1,2,3 : transverse momentum of the leptons
        -pt_MET : norm of missing transverse momentum 
        -phi_1,2,3 : azimuthal angle of the leptons
        -phi_MET : azimuthal angle of the missing transverse momentum 
        -eta_1,2,3 : pseudorapidity of the leptons 
        -eta_1,2,3 : mass of the leptons 
    Output:
        -All combinations of deltaphi between 1 object and the sum of two others
    """
    n = len(pt_1)
    eta_MET = []
    mass_MET = []
    if type(pt_1) == list:
        for i in range(n):
            eta_MET.append(0)
            mass_MET.append(0)
    else:
        eta_MET = zeros_like(pt_1)
        mass_MET = zeros_like(pt_1)
    vector_MET = arr({"pt" : pt_MET,
                      "phi" : phi_MET,
                      "eta" : eta_MET,
                      "M" : mass_MET})
    vector_1 = arr({"pt" : pt_1,
                      "phi" : phi_1,
                      "eta" : eta_1,
                      "M" : mass_1})
    vector_2 = arr({"pt" : pt_2,
                      "phi" : phi_2,
                      "eta" : eta_2,
                      "M" : mass_2})
    vector_3 = arr({"pt" : pt_3,
                      "phi" : phi_3,
                      "eta" : eta_3,
                      "M" : mass_3})
    vectors = [vector_1, vector_2, vector_3, vector_MET]

    groups = [[0,1,2], [1,0,2], [2,0,1], [3,0,1], [3,0,2], [3,1,2], [0,1,3], [0,2,3], [1,0,3], [1,2,3], [2,0,3], [2,1,3]]
    deltaphis = []

    for comb in groups:
        deltaphis.append(deltaphi(vectors[comb[0]].phi, (vectors[comb[1]]+vectors[comb[2]]).phi)) # type: ignore

    return deltaphis

#######################################################

def deltaeta(eta1, eta2):
    """
    Arguments:
        -eta1 : pseudorapidity of the first particle
        -eta2 : pseudorapidity of the second particle    
    """
    # if len(args) == 1:
    #     if len(args[0]) != 2:
    #         raise TypeError("Wrong number of arguments")
    #     eta1 = args[0][0]
    #     eta2 = args[0][1]
    # elif len(args) == 2:
    #     eta1 = args[0]
    #     eta2 = args[1]
    # else:
    #     raise TypeError("Wrong number of arguments")
    return abs(eta1-eta2)

#######################################################

def deltaeta3(pt_1, pt_2, pt_3,
                phi_1, phi_2, phi_3,
                eta_1, eta_2, eta_3,
                mass_1, mass_2, mass_3):
    """
    Arguments:
        -pt_1,2,3 : transverse momentum of the leptons
        -phi_1,2,3 : azimuthal angle of the leptons
        -eta_1,2,3 : pseudorapidity of the leptons 
        -eta_1,2,3 : mass of the leptons 
    Output:
        -All combinations of deltaeta between 1 lepton and the sum of two others
    """
    vector_1 = arr({"pt" : pt_1,
                      "phi" : phi_1,
                      "eta" : eta_1,
                      "M" : mass_1})
    vector_2 = arr({"pt" : pt_2,
                      "phi" : phi_2,
                      "eta" : eta_2,
                      "M" : mass_2})
    vector_3 = arr({"pt" : pt_3,
                      "phi" : phi_3,
                      "eta" : eta_3,
                      "M" : mass_3})
    vectors = [vector_1, vector_2, vector_3]

    groups = [[0,1,2], [1,0,2], [2,0,1]]
    deltaetas = []

    for comb in groups:
        deltaetas.append(deltaeta(vectors[comb[0]].eta, (vectors[comb[1]]+vectors[comb[2]]).eta)) # type: ignore

    return deltaetas

#######################################################

def deltaR(eta1, eta2, phi1, phi2):
    """
    Arguments:
        -eta1 : pseudorapidity of the first particle
        -eta2 : pseudorapidity of the second particle
        -phi1 : azimuthal angle of the first particle
        -phi2 : azimuthal angle of the second particle
    """
    # if len(args) == 1:
    #     if len(args[0]) != 4:
    #         raise TypeError("Wrong number of arguments")
    #     eta1 = args[0][0]
    #     eta2 = args[0][1]
    #     phi1 = args[0][2]
    #     phi2 = args[0][3]
    # elif len(args) == 4:
    #     eta1 = args[0]
    #     eta2 = args[1]
    #     phi1 = args[2]
    #     phi2 = args[3]
    # else:
    #     raise TypeError("Wrong number of arguments")
    return sqrt(deltaeta(eta1, eta2)**2+deltaphi(phi1, phi2)**2) # type: ignore

#######################################################

def deltaR3(pt_1, pt_2, pt_3,
                phi_1, phi_2, phi_3,
                eta_1, eta_2, eta_3,
                mass_1, mass_2, mass_3):
    """
    Arguments:
        -pt_1,2,3 : transverse momentum of the leptons
        -phi_1,2,3 : azimuthal angle of the leptons
        -eta_1,2,3 : pseudorapidity of the leptons 
        -eta_1,2,3 : mass of the leptons 
    Output:
        -All combinations of deltaR between 1 lepton and the sum of two others
    """
    vector_1 = arr({"pt" : pt_1,
                      "phi" : phi_1,
                      "eta" : eta_1,
                      "M" : mass_1})
    vector_2 = arr({"pt" : pt_2,
                      "phi" : phi_2,
                      "eta" : eta_2,
                      "M" : mass_2})
    vector_3 = arr({"pt" : pt_3,
                      "phi" : phi_3,
                      "eta" : eta_3,
                      "M" : mass_3})
    vectors = [vector_1, vector_2, vector_3]

    groups = [[0,1,2], [1,0,2], [2,0,1]]
    deltaRs = []

    for comb in groups:
        vec_sum = vectors[comb[1]]+vectors[comb[2]]  # type: ignore
        deltaRs.append(deltaR(vectors[comb[0]].eta, vec_sum.eta, vectors[comb[0]].phi, vec_sum.phi))  # type: ignore

    return deltaRs

#######################################################

def sum_pt(pts, phis, etas, masses):
    """
    Aguments : 
        -pts : transverse momentum of the particles
        -phis : azimuthal angles of the particles
        -etas : pseudorapidity of the particles 
        -masses : masses of the particles 
    All arguments have 2 coordinates :
        -the first component corresponds to the type of particle (muon, tau, MET)
        -The second coordinate corresponds to the event.
    Output : 
        -Sum of the transverse momentum of the three leptons and the MET
    """
    # if len(args) == 1:
    #     if len(args[0]) != 4:
    #         raise TypeError("Wrong number of arguments")
    #     pts = args[0][0]
    #     phis = args[0][1]
    #     etas = args[0][2]
    #     masses = args[0][3]
    # elif len(args) == 4:
    #     pts = args[0]
    #     phis = args[1]
    #     etas = args[2]
    #     masses = args[3]
    # else:
    #     raise TypeError("Wrong number of arguments")
    p_tot = arr({"pt" : pts[0],
                "phi" : phis[0],
                "eta" : etas[0],
                "M" : masses[0]})
    for i in range(1,len(masses)):
        p_tot += arr({"pt" : pts[i],
                    "phi" : phis[i],
                    "eta" : etas[i],
                    "M" : masses[i]})   # type: ignore
    return p_tot.pt # type: ignore

#######################################################

def transverse_mass(pt_1, pt_2, phi_1, phi_2):
    """
    Arguments : 
        -pt_1 : transverse momentum of the first particle
        -pt_2 : transverse momentum of the second particle
        -phi_1 : azimuthal angle of the first particle
        -phi_2 : azimuthal angle of the second particle
    """
    # if len(args) == 1:
    #     if len(args[0]) != 4:
    #         raise TypeError("Wrong number of arguments")
    #     pt_1 = args[0][0]
    #     pt_2 = args[0][1]
    #     phi_1 = args[0][2]
    #     phi_2 = args[0][3]
    # elif len(args) == 4:
    #     pt_1 = args[0]
    #     pt_2 = args[1]
    #     phi_1 = args[2]
    #     phi_2 = args[3]
    # else:
    #     raise TypeError("Wrong number of arguments")
    return sqrt(2.0*pt_1*pt_2*(1.0 - cos(phi_1-phi_2)))

#######################################################

def transverse_mass3(pt_1, pt_2, pt_3, pt_MET,
                phi_1, phi_2, phi_3, phi_MET,
                eta_1, eta_2, eta_3,
                mass_1, mass_2, mass_3):
    """
    Arguments:
        -pt_1,2,3 : transverse momentum of the leptons
        -pt_MET : norm of missing transverse momentum 
        -phi_1,2,3 : azimuthal angle of the leptons
        -phi_MET : azimuthal angle of the missing transverse momentum 
        -eta_1,2,3 : pseudorapidity of the leptons 
        -eta_1,2,3 : mass of the leptons 
    Output:
        -All combinations of transverse masses between 1 object and the sum of two others
    """
    n = len(pt_1)
    eta_MET = []
    mass_MET = []
    if type(pt_1) == list:
        for i in range(n):
            eta_MET.append(0)
            mass_MET.append(0)
    else:
        eta_MET = zeros_like(pt_1)
        mass_MET = zeros_like(pt_1)
    vector_MET = arr({"pt" : pt_MET,
                      "phi" : phi_MET,
                      "eta" : eta_MET,
                      "M" : mass_MET})
    vector_1 = arr({"pt" : pt_1,
                      "phi" : phi_1,
                      "eta" : eta_1,
                      "M" : mass_1})
    vector_2 = arr({"pt" : pt_2,
                      "phi" : phi_2,
                      "eta" : eta_2,
                      "M" : mass_2})
    vector_3 = arr({"pt" : pt_3,
                      "phi" : phi_3,
                      "eta" : eta_3,
                      "M" : mass_3})
    vectors = [vector_1, vector_2, vector_3, vector_MET]

    groups = [[0,1,2], [1,0,2], [2,0,1], [3,0,1], [3,0,2], [3,1,2], [0,1,3], [0,2,3], [1,0,3], [1,2,3], [2,0,3], [2,1,3]]
    transverse_masses = []

    for comb in groups:
        vec_sum = vectors[comb[0]] + vectors[comb[1]]   # type: ignore
        transverse_masses.append(transverse_mass(vectors[comb[0]].pt, vec_sum.pt, vectors[comb[0]].phi, vec_sum.phi))   # type: ignore

    return transverse_masses

#######################################################

def total_transverse_mass(pt_1, pt_2, pt_3, pt_miss, phi_1, phi_2, phi_3, phi_miss):
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
    # if len(args) == 1:
    #     if len(args[0]) != 8:
    #         raise TypeError("Wrong number of arguments")
    #     pt_1 = args[0][0]
    #     pt_2 = args[0][1]
    #     pt_3 = args[0][2]
    #     pt_miss = args[0][3]
    #     phi_1 = args[0][4]
    #     phi_2 = args[0][5]
    #     phi_3 = args[0][6]
    #     phi_miss = args[0][7]
    # elif len(args) == 8:
    #     pt_1 = args[0]
    #     pt_2 = args[1]
    #     pt_3 = args[2]
    #     pt_miss = args[3]
    #     phi_1 = args[4]
    #     phi_2 = args[5]
    #     phi_3 = args[6]
    #     phi_miss = args[7]
    # else:
    #     raise TypeError("Wrong number of arguments")
    return sqrt(transverse_mass(pt_1, pt_2, phi_1, phi_2)**2 + transverse_mass(pt_1, pt_3, phi_1, phi_3)**2 + 
                transverse_mass(pt_2, pt_3, phi_2, phi_3)**2 + transverse_mass(pt_1, pt_miss, phi_1, phi_miss)**2 +
                transverse_mass(pt_2, pt_miss, phi_2, phi_miss)**2 + transverse_mass(pt_3, pt_miss, phi_3, phi_miss)**2)

#######################################################

def invariant_mass(pts, phis, etas, masses):
    """
    Aguments : 
        -pts : transverse momentum of the particles
        -phis : azimuthal angles of the particles
        -etas : pseudorapidity of the particles 
        -masses : masses of the particles 
    All arguments have 2 coordinates :
        -the first component corresponds to the type of particle (muon, tau, MET)
        -The second coordinate corresponds to the event.
    Output : invariant mass of the sum of the 4-vectors of all objects passed to the function
    """
    p_tot = arr({"pt" : pts[0],
                "phi" : phis[0],
                "eta" : etas[0],
                "M" : masses[0]})
    for i in range(1,len(masses)):
        p_tot += arr({"pt" : pts[i],
                    "phi" : phis[i],
                    "eta" : etas[i],
                    "M" : masses[i]})   # type: ignore
    return p_tot.mass   # type: ignore

#######################################################
    
def HNL_CM_angles_with_MET(charge_1, charge_2, charge_3,
                           pt_1, pt_2, pt_3, pt_MET,
                           phi_1, phi_2, phi_3, phi_MET,
                           eta_1, eta_2, eta_3,
                           mass_1, mass_2, mass_3):
    """
    Arguments : 
        -charge_1,2,3 : charge of the three leptons
        -pt_1,2,3,MET : transverse momentum of the three leptons and the missing momentum
        -phi_1,2,3,MET : azimuthal angle of the three leptons
        -eta_1,2,3 : pseudorapidity of the three leptons
        -mass_1,2,3 : mass of the three leptons
    Output :
        -[angle1, angle2] : angles between 2 leptons of opposite sign (candidate for HNL desintegration) in their rest frame.
                            Since the total charge of the 3 leptons is +/-1, there's 2 choice of lepton pair -> 2 angles in the output
    """
    # if len(args) == 1:
    #     if len(args[0]) != 17:
    #         raise TypeError("Wrong number of arguments")
    #     charge_1 = args[0][0]
    #     charge_2 = args[0][1]
    #     charge_3 = args[0][2]
    #     pt_1 = args[0][3]
    #     pt_2 = args[0][4]
    #     pt_3 = args[0][5]
    #     pt_MET = args[0][6]
    #     phi_1 = args[0][7]
    #     phi_2 = args[0][8]
    #     phi_3 = args[0][9]
    #     phi_MET = args[0][10]
    #     eta_1 = args[0][11]
    #     eta_2 = args[0][12]
    #     eta_3 = args[0][13]
    #     mass_1 = args[0][14]
    #     mass_2 = args[0][15]
    #     mass_3 = args[0][16]
    # elif len(args) == 17:
    #     charge_1 = args[0]
    #     charge_2 = args[1]
    #     charge_3 = args[2]
    #     pt_1 = args[3]
    #     pt_2 = args[4]
    #     pt_3 = args[5]
    #     pt_MET = args[6]
    #     phi_1 = args[7]
    #     phi_2 = args[8]
    #     phi_3 = args[9]
    #     phi_MET = args[10]
    #     eta_1 = args[11]
    #     eta_2 = args[12]
    #     eta_3 = args[13]
    #     mass_1 = args[14]
    #     mass_2 = args[15]
    #     mass_3 = args[16]
    # else:
    #     raise TypeError("Wrong number of arguments")
    
    n = len(charge_1)
    
    eta_MET = [] 
    mass_MET = []
    if type(charge_1) == list:
        for i in range(n): #for i in range(n_bis)
            eta_MET.append(0)
            mass_MET.append(0)
    else:
        eta_MET = zeros_like(charge_1) #zeros((n_bis,))
        mass_MET = zeros_like(charge_1) #zeros((n_bis,))

    vector_MET = arr({"pt" : pt_MET,
                      "phi" : phi_MET,
                      "eta" : eta_MET,
                      "M" : mass_MET})
    
    indices = [1,2,3]

    pts = dict(zip(indices, [[], [], []]))
    phis = dict(zip(indices, [[], [], []]))
    etas = dict(zip(indices, [[], [], []]))
    masses = dict(zip(indices, [[], [], []]))

    for i in range(len(charge_1)):
        charge_tot = charge_1[i] + charge_2[i] + charge_3[i]
        
        if sign(charge_tot) != charge_1[i]:
            pts[1].append(pt_1[i])
            pts[2].append(pt_2[i])
            pts[3].append(pt_3[i])
            phis[1].append(phi_1[i])
            phis[2].append(phi_2[i])
            phis[3].append(phi_3[i])
            etas[1].append(eta_1[i])
            etas[2].append(eta_2[i])
            etas[3].append(eta_3[i])
            masses[1].append(mass_1[i])
            masses[2].append(mass_2[i])
            masses[3].append(mass_3[i])
        if sign(charge_tot) != charge_2[i]:
            pts[1].append(pt_2[i])
            pts[2].append(pt_1[i])
            pts[3].append(pt_3[i])
            phis[1].append(phi_2[i])
            phis[2].append(phi_1[i])
            phis[3].append(phi_3[i])
            etas[1].append(eta_2[i])
            etas[2].append(eta_1[i])
            etas[3].append(eta_3[i])
            masses[1].append(mass_2[i])
            masses[2].append(mass_1[i])
            masses[3].append(mass_3[i])
        if sign(charge_tot) != charge_3[i]:
            pts[1].append(pt_3[i])
            pts[2].append(pt_1[i])
            pts[3].append(pt_2[i])
            phis[1].append(phi_3[i])
            phis[2].append(phi_1[i])
            phis[3].append(phi_2[i])
            etas[1].append(eta_3[i])
            etas[2].append(eta_1[i])
            etas[3].append(eta_2[i])
            masses[1].append(mass_3[i])
            masses[2].append(mass_1[i])
            masses[3].append(mass_2[i])

    
    angles = []
    pair_candidate = [[1,2],[1,3]]

    for pair in pair_candidate:
        i = pair[0]
        j = pair[1]
        vector_i = arr({"pt" : pts[i],
                        "phi" : phis[i],
                        "eta" : etas[i],
                        "M" : masses[i]})
        vector_j = arr({"pt" : pts[j],
                        "phi" : phis[j],
                        "eta" : etas[j],
                        "M" : masses[j]})
        
        vector_tot = vector_i + vector_j + vector_MET   # type: ignore
        vector_i = vector_i.boostCM_of_p4(vector_tot)   # type: ignore
        vector_j = vector_j.boostCM_of_p4(vector_tot)   # type: ignore
        angle = vector_i.deltaangle(vector_j)
        angles.append(angle)
    return angles

#######################################################   

def W_CM_angles_to_plane(charge_1, charge_2, charge_3,
                    pt_1, pt_2, pt_3,
                    phi_1, phi_2, phi_3,
                    eta_1, eta_2, eta_3,
                    mass_1, mass_2, mass_3):
    """
    Arguments : 
        -charge_1,2,3 : charge of the three leptons
        -pt_1,2,3 : transverse momentum of the three leptons
        -phi_1,2,3 : azimuthal angle of the three leptons
        -eta_1,2,3 : pseudorapidity of the three leptons
        -mass_1,2,3 : mass of the three leptons
    Output :
        -[angle1, angle2] : angles between 1 lepton and the plane formed by 2 other leptons of opposite sign (candidate for HNL 
                            desintegration) in the rest frame of the 3 leptons. Since the total charge of the 3 leptons is +/-1, 
                            there's 2 choice of lepton pair -> 2 angles in the output
    """
    # if len(args) == 1:
    #     if len(args[0]) != 15:
    #         raise TypeError("Wrong number of arguments")
    #     charge_1 = args[0][0]
    #     charge_2 = args[0][1]
    #     charge_3 = args[0][2]
    #     pt_1 = args[0][3]
    #     pt_2 = args[0][4]
    #     pt_3 = args[0][5]
    #     phi_1 = args[0][6]
    #     phi_2 = args[0][7]
    #     phi_3 = args[0][8]
    #     eta_1 = args[0][9]
    #     eta_2 = args[0][10]
    #     eta_3 = args[0][11]
    #     mass_1 = args[0][12]
    #     mass_2 = args[0][13]
    #     mass_3 = args[0][14]
    # elif len(args) == 15:
    #     charge_1 = args[0]
    #     charge_2 = args[1]
    #     charge_3 = args[2]
    #     pt_1 = args[3]
    #     pt_2 = args[4]
    #     pt_3 = args[5]
    #     phi_1 = args[6]
    #     phi_2 = args[7]
    #     phi_3 = args[8]
    #     eta_1 = args[9]
    #     eta_2 = args[10]
    #     eta_3 = args[11]
    #     mass_1 = args[12]
    #     mass_2 = args[13]
    #     mass_3 = args[14]
    # else:
    #     raise TypeError("Wrong number of arguments")
    
    indices = [1,2,3]

    pts = dict(zip(indices, [[], [], []]))
    phis = dict(zip(indices, [[], [], []]))
    etas = dict(zip(indices, [[], [], []]))
    masses = dict(zip(indices, [[], [], []]))

    for i in range(len(charge_1)):
        charge_tot = charge_1[i] + charge_2[i] + charge_3[i]
        if sign(charge_tot) != charge_1[i]:
            pts[1].append(pt_1[i])
            pts[2].append(pt_2[i])
            pts[3].append(pt_3[i])
            phis[1].append(phi_1[i])
            phis[2].append(phi_2[i])
            phis[3].append(phi_3[i])
            etas[1].append(eta_1[i])
            etas[2].append(eta_2[i])
            etas[3].append(eta_3[i])
            masses[1].append(mass_1[i])
            masses[2].append(mass_2[i])
            masses[3].append(mass_3[i])
        if sign(charge_tot) != charge_2[i]:
            pts[1].append(pt_2[i])
            pts[2].append(pt_1[i])
            pts[3].append(pt_3[i])
            phis[1].append(phi_2[i])
            phis[2].append(phi_1[i])
            phis[3].append(phi_3[i])
            etas[1].append(eta_2[i])
            etas[2].append(eta_1[i])
            etas[3].append(eta_3[i])
            masses[1].append(mass_2[i])
            masses[2].append(mass_1[i])
            masses[3].append(mass_3[i])
        if sign(charge_tot) != charge_3[i]:
            pts[1].append(pt_3[i])
            pts[2].append(pt_1[i])
            pts[3].append(pt_2[i])
            phis[1].append(phi_3[i])
            phis[2].append(phi_1[i])
            phis[3].append(phi_2[i])
            etas[1].append(eta_3[i])
            etas[2].append(eta_1[i])
            etas[3].append(eta_2[i])
            masses[1].append(mass_3[i])
            masses[2].append(mass_1[i])
            masses[3].append(mass_2[i])

    
    angles = []
    pair_candidate = [[1,2],[1,3]]

    for pair in pair_candidate:
        i = pair[0]
        j = pair[1]
        k = next(value for value in [1, 2, 3] if value != i and value != j)

        vector_first = arr({"pt" : pts[k],
                        "phi" : phis[k],
                        "eta" : etas[k],
                        "M" : masses[k]})
        vector_i = arr({"pt" : pts[i],
                        "phi" : phis[i],
                        "eta" : etas[i],
                        "M" : masses[i]})
        vector_j = arr({"pt" : pts[j],
                        "phi" : phis[j],
                        "eta" : etas[j],
                        "M" : masses[j]})
        vector_tot = vector_i + vector_j + vector_first # type: ignore
        vector_i = vector_i.boostCM_of_p4(vector_tot)   # type: ignore
        vector_j = vector_j.boostCM_of_p4(vector_tot)   # type: ignore
        vector_first = vector_first.boostCM_of_p4(vector_tot)   # type: ignore
        normal = vector_i.cross(vector_j)
        angle = vector_first.deltaangle(normal)
        angles.append(abs(pi/2-angle))
    return angles
    
#######################################################   

def W_CM_angles_to_plane_with_MET(charge_1, charge_2, charge_3,
                           pt_1, pt_2, pt_3, pt_MET,
                           phi_1, phi_2, phi_3, phi_MET,
                           eta_1, eta_2, eta_3,
                           mass_1, mass_2, mass_3):
    """
    Arguments : 
        -charge_1,2,3 : charge of the three leptons
        -pt_1,2,3,MET : transverse momentum of the three leptons and the missing momentum
        -phi_1,2,3,MET : azimuthal angle of the three leptons
        -eta_1,2,3 : pseudorapidity of the three leptons
        -mass_1,2,3 : mass of the three leptons
    Output :
        -[angle1, angle2] : angles between 1 lepton and the plane formed by 2 other leptons of opposite sign (candidate for HNL 
                            desintegration) in the rest frame of the 3 leptons. Since the total charge of the 3 leptons is +/-1, 
                            there's 2 choice of lepton pair -> 2 angles in the output
    """
    # if len(args) == 1:
    #     if len(args[0]) != 17:
    #         raise TypeError("Wrong number of arguments")
    #     charge_1 = args[0][0]
    #     charge_2 = args[0][1]
    #     charge_3 = args[0][2]
    #     pt_1 = args[0][3]
    #     pt_2 = args[0][4]
    #     pt_3 = args[0][5]
    #     pt_MET = args[0][6]
    #     phi_1 = args[0][7]
    #     phi_2 = args[0][8]
    #     phi_3 = args[0][9]
    #     phi_MET = args[0][10]
    #     eta_1 = args[0][11]
    #     eta_2 = args[0][12]
    #     eta_3 = args[0][13]
    #     mass_1 = args[0][14]
    #     mass_2 = args[0][15]
    #     mass_3 = args[0][16]
    # elif len(args) == 17:
    #     charge_1 = args[0]
    #     charge_2 = args[1]
    #     charge_3 = args[2]
    #     pt_1 = args[3]
    #     pt_2 = args[4]
    #     pt_3 = args[5]
    #     pt_MET = args[6]
    #     phi_1 = args[7]
    #     phi_2 = args[8]
    #     phi_3 = args[9]
    #     phi_MET = args[10]
    #     eta_1 = args[11]
    #     eta_2 = args[12]
    #     eta_3 = args[13]
    #     mass_1 = args[14]
    #     mass_2 = args[15]
    #     mass_3 = args[16]
    # else:
    #     raise TypeError("Wrong number of arguments")
    
    n = len(charge_1)
    eta_MET = []
    mass_MET = []
    if type(charge_1) == list:
        for i in range(n):
            eta_MET.append(0)
            mass_MET.append(0)
    else:
        eta_MET = zeros_like(charge_1)
        mass_MET = zeros_like(charge_1)
    vector_MET = arr({"pt" : pt_MET,
                      "phi" : phi_MET,
                      "eta" : eta_MET,
                      "M" : mass_MET})
    
    indices = [1,2,3]

    pts = dict(zip(indices, [[], [], []]))
    phis = dict(zip(indices, [[], [], []]))
    etas = dict(zip(indices, [[], [], []]))
    masses = dict(zip(indices, [[], [], []]))

    for i in range(len(charge_1)):
        charge_tot = charge_1[i] + charge_2[i] + charge_3[i]
        if sign(charge_tot) != charge_1[i]:
            pts[1].append(pt_1[i])
            pts[2].append(pt_2[i])
            pts[3].append(pt_3[i])
            phis[1].append(phi_1[i])
            phis[2].append(phi_2[i])
            phis[3].append(phi_3[i])
            etas[1].append(eta_1[i])
            etas[2].append(eta_2[i])
            etas[3].append(eta_3[i])
            masses[1].append(mass_1[i])
            masses[2].append(mass_2[i])
            masses[3].append(mass_3[i])
        if sign(charge_tot) != charge_2[i]:
            pts[1].append(pt_2[i])
            pts[2].append(pt_1[i])
            pts[3].append(pt_3[i])
            phis[1].append(phi_2[i])
            phis[2].append(phi_1[i])
            phis[3].append(phi_3[i])
            etas[1].append(eta_2[i])
            etas[2].append(eta_1[i])
            etas[3].append(eta_3[i])
            masses[1].append(mass_2[i])
            masses[2].append(mass_1[i])
            masses[3].append(mass_3[i])
        if sign(charge_tot) != charge_3[i]:
            pts[1].append(pt_3[i])
            pts[2].append(pt_1[i])
            pts[3].append(pt_2[i])
            phis[1].append(phi_3[i])
            phis[2].append(phi_1[i])
            phis[3].append(phi_2[i])
            etas[1].append(eta_3[i])
            etas[2].append(eta_1[i])
            etas[3].append(eta_2[i])
            masses[1].append(mass_3[i])
            masses[2].append(mass_1[i])
            masses[3].append(mass_2[i])

    
    angles = []
    pair_candidate = [[1,2],[1,3]]

    for pair in pair_candidate:
        i = pair[0]
        j = pair[1]
        k = next(value for value in [1, 2, 3] if value != i and value != j)

        vector_first = arr({"pt" : pts[k],
                        "phi" : phis[k],
                        "eta" : etas[k],
                        "M" : masses[k]})
        vector_i = arr({"pt" : pts[i],
                        "phi" : phis[i],
                        "eta" : etas[i],
                        "M" : masses[i]})
        vector_j = arr({"pt" : pts[j],
                        "phi" : phis[j],
                        "eta" : etas[j],
                        "M" : masses[j]})
        vector_tot = vector_i + vector_j + vector_first + vector_MET    # type: ignore
        vector_i = vector_i.boostCM_of_p4(vector_tot)   # type: ignore
        vector_j = vector_j.boostCM_of_p4(vector_tot)   # type: ignore
        vector_first = vector_first.boostCM_of_p4(vector_tot)   # type: ignore
        normal = vector_i.cross(vector_j)
        angle = vector_first.deltaangle(normal)
        angles.append(abs(pi/2-angle))
    return angles

####################################################### 

def W_CM_angles(pt_1, pt_2, pt_3, pt_MET,
                phi_1, phi_2, phi_3, phi_MET,
                eta_1, eta_2, eta_3,
                mass_1, mass_2, mass_3):
    """
    Arguments : 
        -pt_1,2,3,MET : transverse momentum of the three leptons and the missing momentum
        -phi_1,2,3,MET : azimuthal angle of the three leptons
        -eta_1,2,3 : pseudorapidity of the three leptons
        -mass_1,2,3 : mass of the three leptons
    Output :
        -Angle between the momenta of 2 objects in the center of mass frame of the first W boson (without considering missing momentum),
         for all 6 combination of 2 objects
    """

    n = len(pt_1)
    eta_MET = []
    mass_MET = []
    if type(pt_1) == list:
        for i in range(n):
            eta_MET.append(0)
            mass_MET.append(0)
    else:
        eta_MET = zeros_like(pt_1)
        mass_MET = zeros_like(pt_1)
    vector_MET = arr({"pt" : pt_MET,
                      "phi" : phi_MET,
                      "eta" : eta_MET,
                      "M" : mass_MET})
    vector_1 = arr({"pt" : pt_1,
                      "phi" : phi_1,
                      "eta" : eta_1,
                      "M" : mass_1})
    vector_2 = arr({"pt" : pt_2,
                      "phi" : phi_2,
                      "eta" : eta_2,
                      "M" : mass_2})
    vector_3 = arr({"pt" : pt_3,
                      "phi" : phi_3,
                      "eta" : eta_3,
                      "M" : mass_3})
    
    vectors = [vector_1, vector_2, vector_3, vector_MET]
    
    vector_tot = vector_1 + vector_2 + vector_3 # type: ignore
    
    for i in range(len(vectors)):
        vectors[i] = vectors[i].boostCM_of_p4(vector_tot)   # type: ignore
    
    pairs = [[0,1], [0,2], [1,2], [0,3], [1,3], [2,3]]
    angles = []

    for pair in pairs:
        angle = vectors[pair[0]].deltaangle(vectors[pair[1]])   # type: ignore
        angles.append(angle)
    
    return angles

####################################################### 

def W_CM_angles_with_MET(pt_1, pt_2, pt_3, pt_MET,
                            phi_1, phi_2, phi_3, phi_MET,
                            eta_1, eta_2, eta_3,
                            mass_1, mass_2, mass_3):
    """
    Arguments : 
        -pt_1,2,3,MET : transverse momentum of the three leptons and the missing momentum
        -phi_1,2,3,MET : azimuthal angle of the three leptons
        -eta_1,2,3 : pseudorapidity of the three leptons
        -mass_1,2,3 : mass of the three leptons
    Output :
        -Angle between the momenta of 2 objects in the center of mass frame of the first W boson (with missing momentum),
         for all 6 combination of 2 objects
    """

    n = len(pt_1)
    eta_MET = []
    mass_MET = []
    if type(pt_1) == list:
        for i in range(n):
            eta_MET.append(0)
            mass_MET.append(0)
    else:
        eta_MET = zeros_like(pt_1)
        mass_MET = zeros_like(pt_1)
    vector_MET = arr({"pt" : pt_MET,
                      "phi" : phi_MET,
                      "eta" : eta_MET,
                      "M" : mass_MET})
    vector_1 = arr({"pt" : pt_1,
                      "phi" : phi_1,
                      "eta" : eta_1,
                      "M" : mass_1})
    vector_2 = arr({"pt" : pt_2,
                      "phi" : phi_2,
                      "eta" : eta_2,
                      "M" : mass_2})
    vector_3 = arr({"pt" : pt_3,
                      "phi" : phi_3,
                      "eta" : eta_3,
                      "M" : mass_3})
    
    vectors = [vector_1, vector_2, vector_3, vector_MET]
    
    vector_tot = vector_1 + vector_2 + vector_3 + vector_MET    # type: ignore
    
    for i in range(len(vectors)):
        vectors[i] = vectors[i].boostCM_of_p4(vector_tot)   # type: ignore
    
    pairs = [[0,1], [0,2], [1,2], [0,3], [1,3], [2,3]]
    angles = []

    for pair in pairs:
        angle = vectors[pair[0]].deltaangle(vectors[pair[1]])   # type: ignore
        angles.append(angle)
    
    return angles

####################################################### 

def HNL_CM_masses(charge_1, charge_2, charge_3,
                    pt_1, pt_2, pt_3,
                    phi_1, phi_2, phi_3,
                    eta_1, eta_2, eta_3,
                    mass_1, mass_2, mass_3):
    """
    Arguments : 
        -charge_1,2,3 : charge of the three leptons
        -pt_1,2,3 : transverse momentum of the three leptons
        -phi_1,2,3 : azimuthal angle of the three leptons
        -eta_1,2,3 : pseudorapidity of the three leptons
        -mass_1,2,3 : mass of the three leptons
    Output :
        -[HNL_mass1, HNL_mass2] : invariant mass of the sum of 2 leptons of opposite sign (candidate for HNL desintegration). 
                        Since the total charge of the 3 leptons is +/-1, there's 2 choice of lepton pair -> 2 masses in the output
    """
    # if len(args) == 1:
    #     if len(args[0]) != 15:
    #         raise TypeError("Wrong number of arguments")
    #     charge_1 = args[0][0]
    #     charge_2 = args[0][1]
    #     charge_3 = args[0][2]
    #     pt_1 = args[0][3]
    #     pt_2 = args[0][4]
    #     pt_3 = args[0][5]
    #     phi_1 = args[0][6]
    #     phi_2 = args[0][7]
    #     phi_3 = args[0][8]
    #     eta_1 = args[0][9]
    #     eta_2 = args[0][10]
    #     eta_3 = args[0][11]
    #     mass_1 = args[0][12]
    #     mass_2 = args[0][13]
    #     mass_3 = args[0][14]
    # elif len(args) == 15:
    #     charge_1 = args[0]
    #     charge_2 = args[1]
    #     charge_3 = args[2]
    #     pt_1 = args[3]
    #     pt_2 = args[4]
    #     pt_3 = args[5]
    #     phi_1 = args[6]
    #     phi_2 = args[7]
    #     phi_3 = args[8]
    #     eta_1 = args[9]
    #     eta_2 = args[10]
    #     eta_3 = args[11]
    #     mass_1 = args[12]
    #     mass_2 = args[13]
    #     mass_3 = args[14]
    # else:
    #     raise TypeError("Wrong number of arguments")
    
    indices = [1,2,3]

    pts = dict(zip(indices, [[], [], []]))
    phis = dict(zip(indices, [[], [], []]))
    etas = dict(zip(indices, [[], [], []]))
    masses = dict(zip(indices, [[], [], []]))

    for i in range(len(charge_1)):
        charge_tot = charge_1[i] + charge_2[i] + charge_3[i]
        if sign(charge_tot) != charge_1[i]:
            pts[1].append(pt_1[i])
            pts[2].append(pt_2[i])
            pts[3].append(pt_3[i])
            phis[1].append(phi_1[i])
            phis[2].append(phi_2[i])
            phis[3].append(phi_3[i])
            etas[1].append(eta_1[i])
            etas[2].append(eta_2[i])
            etas[3].append(eta_3[i])
            masses[1].append(mass_1[i])
            masses[2].append(mass_2[i])
            masses[3].append(mass_3[i])
        if sign(charge_tot) != charge_2[i]:
            pts[1].append(pt_2[i])
            pts[2].append(pt_1[i])
            pts[3].append(pt_3[i])
            phis[1].append(phi_2[i])
            phis[2].append(phi_1[i])
            phis[3].append(phi_3[i])
            etas[1].append(eta_2[i])
            etas[2].append(eta_1[i])
            etas[3].append(eta_3[i])
            masses[1].append(mass_2[i])
            masses[2].append(mass_1[i])
            masses[3].append(mass_3[i])
        if sign(charge_tot) != charge_3[i]:
            pts[1].append(pt_3[i])
            pts[2].append(pt_1[i])
            pts[3].append(pt_2[i])
            phis[1].append(phi_3[i])
            phis[2].append(phi_1[i])
            phis[3].append(phi_2[i])
            etas[1].append(eta_3[i])
            etas[2].append(eta_1[i])
            etas[3].append(eta_2[i])
            masses[1].append(mass_3[i])
            masses[2].append(mass_1[i])
            masses[3].append(mass_2[i])

    
    HNL_masses = []
    pair_candidate = [[1,2],[1,3]]

    for pair in pair_candidate:
        i = pair[0]
        j = pair[1]

        vector_i = arr({"pt" : pts[i],
                        "phi" : phis[i],
                        "eta" : etas[i],
                        "M" : masses[i]})
        vector_j = arr({"pt" : pts[j],
                        "phi" : phis[j],
                        "eta" : etas[j],
                        "M" : masses[j]})
        vector_tot = vector_i + vector_j    # type: ignore
        HNL_masses.append(vector_tot.mass)
    return HNL_masses

####################################################### 

def HNL_CM_masses_with_MET(charge_1, charge_2, charge_3,
                           pt_1, pt_2, pt_3, pt_MET,
                           phi_1, phi_2, phi_3, phi_MET,
                           eta_1, eta_2, eta_3,
                           mass_1, mass_2, mass_3):
    """
    Arguments : 
        -charge_1,2,3 : charge of the three leptons
        -pt_1,2,3,MET : transverse momentum of the three leptons and the missing momentum
        -phi_1,2,3,MET : azimuthal angle of the three leptons
        -eta_1,2,3 : pseudorapidity of the three leptons
        -mass_1,2,3 : mass of the three leptons
    Output :
        -[HNL_mass1, HNL_mass2] : invariant mass of the sum of 2 leptons of opposite sign (candidate for HNL desintegration). 
                        Since the total charge of the 3 leptons is +/-1, there's 2 choice of lepton pair -> 2 masses in the output
    """
    # if len(args) == 1:
    #     if len(args[0]) != 17:
    #         raise TypeError("Wrong number of arguments")
    #     charge_1 = args[0][0]
    #     charge_2 = args[0][1]
    #     charge_3 = args[0][2]
    #     pt_1 = args[0][3]
    #     pt_2 = args[0][4]
    #     pt_3 = args[0][5]
    #     pt_MET = args[0][6]
    #     phi_1 = args[0][7]
    #     phi_2 = args[0][8]
    #     phi_3 = args[0][9]
    #     phi_MET = args[0][10]
    #     eta_1 = args[0][11]
    #     eta_2 = args[0][12]
    #     eta_3 = args[0][13]
    #     mass_1 = args[0][14]
    #     mass_2 = args[0][15]
    #     mass_3 = args[0][16]
    # elif len(args) == 17:
    #     charge_1 = args[0]
    #     charge_2 = args[1]
    #     charge_3 = args[2]
    #     pt_1 = args[3]
    #     pt_2 = args[4]
    #     pt_3 = args[5]
    #     pt_MET = args[6]
    #     phi_1 = args[7]
    #     phi_2 = args[8]
    #     phi_3 = args[9]
    #     phi_MET = args[10]
    #     eta_1 = args[11]
    #     eta_2 = args[12]
    #     eta_3 = args[13]
    #     mass_1 = args[14]
    #     mass_2 = args[15]
    #     mass_3 = args[16]
    # else:
    #     raise TypeError("Wrong number of arguments")
    
    n = len(charge_1)
    eta_MET = []
    mass_MET = []
    if type(charge_1) == list:
        for i in range(n):
            eta_MET.append(0)
            mass_MET.append(0)
    else:
        eta_MET = zeros_like(charge_1)
        mass_MET = zeros_like(charge_1)
    vector_MET = arr({"pt" : pt_MET,
                      "phi" : phi_MET,
                      "eta" : eta_MET,
                      "M" : mass_MET})
    
    indices = [1,2,3]

    pts = dict(zip(indices, [[], [], []]))
    phis = dict(zip(indices, [[], [], []]))
    etas = dict(zip(indices, [[], [], []]))
    masses = dict(zip(indices, [[], [], []]))

    for i in range(len(charge_1)):
        charge_tot = charge_1[i] + charge_2[i] + charge_3[i]
        if sign(charge_tot) != charge_1[i]:
            pts[1].append(pt_1[i])
            pts[2].append(pt_2[i])
            pts[3].append(pt_3[i])
            phis[1].append(phi_1[i])
            phis[2].append(phi_2[i])
            phis[3].append(phi_3[i])
            etas[1].append(eta_1[i])
            etas[2].append(eta_2[i])
            etas[3].append(eta_3[i])
            masses[1].append(mass_1[i])
            masses[2].append(mass_2[i])
            masses[3].append(mass_3[i])
        if sign(charge_tot) != charge_2[i]:
            pts[1].append(pt_2[i])
            pts[2].append(pt_1[i])
            pts[3].append(pt_3[i])
            phis[1].append(phi_2[i])
            phis[2].append(phi_1[i])
            phis[3].append(phi_3[i])
            etas[1].append(eta_2[i])
            etas[2].append(eta_1[i])
            etas[3].append(eta_3[i])
            masses[1].append(mass_2[i])
            masses[2].append(mass_1[i])
            masses[3].append(mass_3[i])
        if sign(charge_tot) != charge_3[i]:
            pts[1].append(pt_3[i])
            pts[2].append(pt_1[i])
            pts[3].append(pt_2[i])
            phis[1].append(phi_3[i])
            phis[2].append(phi_1[i])
            phis[3].append(phi_2[i])
            etas[1].append(eta_3[i])
            etas[2].append(eta_1[i])
            etas[3].append(eta_2[i])
            masses[1].append(mass_3[i])
            masses[2].append(mass_1[i])
            masses[3].append(mass_2[i])

    
    HNL_masses = []
    pair_candidate = [[1,2],[1,3]]

    for pair in pair_candidate:
        i = pair[0]
        j = pair[1]

        vector_i = arr({"pt" : pts[i],
                        "phi" : phis[i],
                        "eta" : etas[i],
                        "M" : masses[i]})
        vector_j = arr({"pt" : pts[j],
                        "phi" : phis[j],
                        "eta" : etas[j],
                        "M" : masses[j]})
        vector_tot = vector_i + vector_j + vector_MET   # type: ignore
        HNL_masses.append(vector_tot.mass)
    return HNL_masses

####################################################### 



