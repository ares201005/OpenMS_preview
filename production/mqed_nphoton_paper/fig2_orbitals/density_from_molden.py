from pyscf import scf
from pyscf.tools import cubegen, molden


def molden_to_density(molden_path, cube_path):

    # Create objects from molden file
    mol, __, mo_coeff, mo_occ, __, __ = molden.load(molden_path)

    # Create density matrix
    molden_dm = scf.hf.make_rdm1(mo_coeff=mo_coeff, mo_occ=mo_occ)

    # Write density in cube format
    cube = cubegen.density(
        mol=mol,
        outfile=cube_path,
        dm=molden_dm
    )

    return mol, molden_dm, cube


def cube_from_density(mol, cube_path, dm):

    return cubegen.density(
        mol=mol,
        outfile=cube_path,
        dm=dm
    )


if __name__ == "__main__":

    # Basis set
    #basis = "6-31g"
    basis = "cc-pvdz"

    # QED-HF paths
    qedhf_molden_path1 = f"{basis}/qedhf/c3h4o2_0.000.molden"
    qedhf_molden_path2 = f"{basis}/qedhf/c3h4o2_0.150.molden"
    qedhf_molden_path3 = f"{basis}/qedhf/c3h4o2_0.300.molden"

    qedhf_cube_path1 = f"{basis}/qedhf/cube/c3h4o2_0.000.cube"
    qedhf_cube_path2 = f"{basis}/qedhf/cube/c3h4o2_0.150.cube"
    qedhf_cube_path3 = f"{basis}/qedhf/cube/c3h4o2_0.300.cube"

    # SC-QED-HF paths
    scqed_molden_path1 = f"{basis}/scqed/c3h4o2_0.000.molden"
    scqed_molden_path2 = f"{basis}/scqed/c3h4o2_0.150.molden"
    scqed_molden_path3 = f"{basis}/scqed/c3h4o2_0.300.molden"

    scqed_cube_path1 = f"{basis}/scqed/cube/c3h4o2_0.000.cube"
    scqed_cube_path2 = f"{basis}/scqed/cube/c3h4o2_0.150.cube"
    scqed_cube_path3 = f"{basis}/scqed/cube/c3h4o2_0.300.cube"

    # VT-QED-HF paths
    vtqed_molden_path1 = f"{basis}/vtqed/c3h4o2_0.000.molden"
    vtqed_molden_path2 = f"{basis}/vtqed/c3h4o2_0.150.molden"
    vtqed_molden_path3 = f"{basis}/vtqed/c3h4o2_0.300.molden"

    vtqed_cube_path1 = f"{basis}/vtqed/cube/c3h4o2_0.000.cube"
    vtqed_cube_path2 = f"{basis}/vtqed/cube/c3h4o2_0.150.cube"
    vtqed_cube_path3 = f"{basis}/vtqed/cube/c3h4o2_0.300.cube"


    #############


    # Create QED-HF densities
    mol, qedhf_dm1, qedhf_cube1 = molden_to_density(
        molden_path=qedhf_molden_path1,
        cube_path=qedhf_cube_path1
    )

    __, qedhf_dm2, qedhf_cube2 = molden_to_density(
        molden_path=qedhf_molden_path2,
        cube_path=qedhf_cube_path2
    )

    __, qedhf_dm3, qedhf_cube3 = molden_to_density(
        molden_path=qedhf_molden_path3,
        cube_path=qedhf_cube_path3
    )

    # Delta densities; QED-HF
    delta150_dm = qedhf_dm2 - qedhf_dm1
    delta300_dm = qedhf_dm3 - qedhf_dm1
    qedhf_delta150_path = f"{basis}/qedhf/cube/c3h4o2_delta150.cube"
    qedhf_delta300_path = f"{basis}/qedhf/cube/c3h4o2_delta300.cube"

    qedhf_delta150_cube = cube_from_density(
        mol=mol,
        cube_path=qedhf_delta150_path,
        dm=delta150_dm
    )

    qedhf_delta300_cube = cube_from_density(
        mol=mol,
        cube_path=qedhf_delta300_path,
        dm=delta300_dm
    )


    #############


    # Create SC-QED-HF densities
    mol, scqed_dm1, scqed_cube1 = molden_to_density(
        molden_path=scqed_molden_path1,
        cube_path=scqed_cube_path1
    )

    mol, scqed_dm2, scqed_cube2 = molden_to_density(
        molden_path=scqed_molden_path2,
        cube_path=scqed_cube_path2
    )

    mol, scqed_dm3, scqed_cube3 = molden_to_density(
        molden_path=scqed_molden_path3,
        cube_path=scqed_cube_path3
    )

    # Delta densities; SC-QED-HF
    sc_delta150_dm = scqed_dm2 - scqed_dm1
    sc_delta300_dm = scqed_dm3 - scqed_dm1
    scqed_delta150_path = f"{basis}/scqed/cube/scqed_c3h4o2_delta150.cube"
    scqed_delta300_path = f"{basis}/scqed/cube/scqed_c3h4o2_delta300.cube"

    scqed_delta150_cube = cube_from_density(
        mol=mol,
        cube_path=scqed_delta150_path,
        dm=sc_delta150_dm
    )

    scqed_delta300_cube = cube_from_density(
        mol=mol,
        cube_path=scqed_delta300_path,
        dm=sc_delta300_dm
    )


    #############


    # Create VT-QED-HF densities
    mol, vtqed_dm1, vtqed_cube1 = molden_to_density(
        molden_path=vtqed_molden_path1,
        cube_path=vtqed_cube_path1
    )

    mol, vtqed_dm2, vtqed_cube2 = molden_to_density(
        molden_path=vtqed_molden_path2,
        cube_path=vtqed_cube_path2
    )

    mol, vtqed_dm3, vtqed_cube3 = molden_to_density(
        molden_path=vtqed_molden_path3,
        cube_path=vtqed_cube_path3
    )

    # Delta densities; VT-QED-HF
    vt_delta150_dm = vtqed_dm2 - vtqed_dm1
    vt_delta300_dm = vtqed_dm3 - vtqed_dm1
    vtqed_delta150_path = f"{basis}/vtqed/cube/vtqed_c3h4o2_delta150.cube"
    vtqed_delta300_path = f"{basis}/vtqed/cube/vtqed_c3h4o2_delta300.cube"

    vtqed_delta150_cube = cube_from_density(
        mol=mol,
        cube_path=vtqed_delta150_path,
        dm=vt_delta150_dm
    )

    vtqed_delta300_cube = cube_from_density(
        mol=mol,
        cube_path=vtqed_delta300_path,
        dm=vt_delta300_dm
    )
