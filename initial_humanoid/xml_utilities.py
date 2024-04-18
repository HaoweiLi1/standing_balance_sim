# calculate mass, geometry, and controller gain data from literature equations
def calculate_kp_and_geom(weight, height):
    
    M_total = weight # [kg]
    H_total = height # [meters]
    m_feet = 2*0.0145 * M_total
    m_body = M_total - m_feet
    l_COM = 0.575*H_total
    l_foot = 0.152*H_total
    h_f = 0.039*H_total
    a = 0.19*l_foot
    K_p = m_body * 9.81 * l_COM

    return h_f, m_feet, m_body, l_COM, l_foot, a, K_p


def set_geometry_params(root, m_feet, m_body, l_COM, l_foot, a, H_total, h_f, trans_fric, roll_fric):

    for geom in root.iter('geom'):
        if geom.get('name') == "long_link_geom":
            geom.set('mass', "0")
            geom.set('fromto', f'0 0 {H_total} 0 0 0') # this from (x,y,z)_1 to (x,y,z)_2 is w.r.t the long_link_body frame
            # geom.set('pos', f'0 0 {H_total-l_COM}')

        elif geom.get('name') == "m_body":    
            geom.set('mass', str(m_body))
            geom.set('pos', f"0 0 {l_COM}") # this x,y,z position is w.r.t the long_link_body frame
            # geom.set('size', 0.05)
        
        elif geom.get('name') == "foot":
            geom.set('pos', f'0 0 0') # this x,y,z position is w.r.t the foot_body frame

    for body in root.iter('body'):
            if body.get('name') == "foot":
                # poo = body.get('pos')
                # print(f'pos: {poo}')
                body.set('pos',  f'0 0 0.0') # this x,y,z position is w.r.t the global frame

            elif body.get('name') == "long_link_body":
                # size = float(body.get('size'))
                body.set('pos', f'{l_foot/2-a} 0 {h_f}') # this x,y,z position is w.r.t the foot reference frame

    for joint in root.iter('joint'):
            if joint.get('name') == "ankle_hinge":
                joint.set("pos", f"0 0 0") # this x,y,z position is w.r.t the long_link_body reference frame

            elif joint.get('name') == "rotation_dof":
                joint.set('pos', f'{l_foot/2-a} 0 {h_f}') # this x,y,z position is w.r.t the foot_body reference frame (aligns with world frame)

            elif joint.get('name') == "joint_slide_x":
                joint.set('pos', f"{l_foot/2-a} 0 0.035") # this x,y,z position is w.r.t the foot_body reference frame (aligns with world frame)

            elif joint.get('name') == "joint_slide_z":
                joint.set('pos', f"{l_foot/2-a} 0 0.035") # this x,y,z position is w.r.t the foot_body reference frame (aligns with world frame)

    for pair in root.iter('pair'):
        if pair.get('name') == "foot_ground_friction":
            pair.set('friction', f"{trans_fric} {trans_fric} 0.99 {roll_fric} {roll_fric}")

    for mesh in root.iter('mesh'):
        if mesh.get('name') == "foot_mesh":
            mesh.set('vertex', f"{-l_foot/2} -0.045 0   {-l_foot/2} 0.045 0   {l_foot/2} -0.045 0   {l_foot/2} 0.045 0  {l_foot/2-a} -0.045 {h_f} {l_foot/2-a} 0.045 {h_f}")

    for site in root.iter('site'):
        if site.get('name') == "front_foot_site":
            # site.set('pos', f"{-l_foot/2} 0 0.05")
            site.set('fromto', f"{-l_foot/2} 0 0.0 {-l_foot/2} 0 0.1")
            # site.set('fromto', f"0 0 0.0 0 0 0.1")

        elif site.get('name') == "back_foot_site":
            # site.set('pos', f"{l_foot/2} 0 0.05")
            site.set('fromto', f"{l_foot/2} 0 0.0 {l_foot/2} 0 0.1")
    # for opt in root.iter('option'):
    #     if opt.get('name') == "gravity":
    #         opt.set('')
            # geom.set('mass', str(m_feet))
            # mesh.set('vertex', f"{-l_foot/2} 0 0  {l_foot/2} 0 0  0 -0.035 0  0 0.035 0  {l_foot/2-a} 0 {h_f}")
