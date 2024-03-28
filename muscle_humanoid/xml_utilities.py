# calculate mass, geometry, and controller gain data from literature equations
def calculate_kp_and_geom(weight, height):
    
    M_total = weight # [kg]
    H_total = height # [meters]
    m_feet = 2 * 0.0145 * M_total
    m_body = M_total - m_feet
    l_COM = 0.575*H_total
    l_foot = 0.152*H_total
    a = 0.19*l_foot
    K_p = m_body * 9.81 * l_COM

    return m_feet, m_body, l_COM, l_foot, a, K_p


def set_geometry_params(root, m_feet, m_body, l_COM, l_foot, a, H_total, h_f, trans_fric, roll_fric):

    # modify properties of various geoms in the model
    for geom in root.iter('geom'):
        if geom.get('name') == "shin_geom":
            geom.set('mass', "0")
            geom.set('fromto', f'0 0 {H_total} 0 0 0')
            # geom.set('pos', f'0 0 {H_total-l_COM}')

        elif geom.get('name') == "m_body":
            geom.set('mass', str(m_body))
            # geom.set('size', 0.05)

        # elif geom.get('name') == "foot1_right":

        #     geom.set('fromto', f'0 .02 0 {l_foot} .02 0')
        #     geom.set('mass', str(m_feet))
        
        # elif geom.get('name') == "foot":
        #     geom.set('pos', f'{0} 0')

    # modify the mesh which composes the foot geom in this model
    for mesh in root.iter('mesh'):
        if mesh.get('name') == "foot_mesh":
            mesh.set('vertex', f"{-l_foot/2} 0 0  {l_foot} 0 0  0 -0.035 0  0 0.035 0  {l_foot/2-a} 0 {h_f}")

    # modify properties of various bodies in the model
    for body in root.iter('body'):
            if body.get('name') == "foot":
                # poo = body.get('pos')
                # print(f'pos: {poo}')
                body.set('pos',  f'0 0 0.1')

            elif body.get('name') == "shin_body":
                # size = float(body.get('size'))
                body.set('pos', f'{l_foot/2-a} 0 {h_f}')
                # body.set('inertial', f"{l_foot/2-a} 0 {h_f+l_COM}")

    # modify properties of various joints in the model
    for joint in root.iter('joint'):
            if joint.get('name') == "ankle_hinge":
                joint.set("pos", f"0 0 0")

            elif joint.get('name') == "rotation_dof":
                joint.set('pos', f'-{l_foot} 0 0.035')

            elif joint.get('name') == "joint_slide_x":
                joint.set('pos', f"{l_foot/2} 0 0.035")

            elif joint.get('name') == "joint_slide_z":
                joint.set('pos', f"{l_foot/2} 0 0.035")

    # modify the foot-ground contact friction parameters
    for pair in root.iter('pair'):
        if pair.get('name') == "foot_ground_friction":
            pair.set('friction', f"{trans_fric} {trans_fric} 0.99 {roll_fric} {roll_fric}")

    
    # modify properties of the sites, which are used as anchor points that 
    # attach the tendons in our model to the geoms in our model
    for site in root.iter('site'):
        if site.get('name') == "front_foot_site":
            site.set('pos', f"{-l_foot/3} 0 0.05")
        
        elif site.get('name') == "front_shin_site":
            site.set('pos', f"-0.0125 0 {3*h_f}")

        elif site.get('name') == "back_foot_site":
            site.set('pos', f"{l_foot} 0 0.05")

        elif site.get('name') == "back_shin_site":
            site.set('pos', f"0.0125 0 {3*h_f}")

    # modify properties of the sites that the tendons are connected to
    # we must iterate thru the "spatial" tendons because this is the
    # type of tendon I used in the XML file
    for tendie in root.iter('spatial'):
        if tendie.get('name') == "front_tendon":
            pass
            