import xml.etree.ElementTree as ET

# calculate mass, geometry, and controller gain data from literature equations
# def calculate_kp_and_geom(weight, height):
    
#     M_total = weight # [kg]
#     H_total = height # [meters]
#     m_feet = 2*0.0145 * M_total
#     m_body = M_total - m_feet
#     l_COM = 0.575*H_total
#     l_foot = 0.152*H_total
#     h_f = 0.039*H_total
#     a = 0.19*l_foot
#     K_p = m_body * 9.81 * l_COM 

#     return h_f, m_feet, m_body, l_COM, l_foot, a, K_p
def calculate_kp_and_geom(weight, height):
    
    M_total = weight # [kg]
    H_total = height # [meters]
    
    # Original parameters
    m_feet = 2*0.0145 * M_total
    m_body = M_total - m_feet
    l_COM = 0.575*H_total
    l_foot = 0.152*H_total
    h_f = 0.039*H_total
    a = 0.19*l_foot
    K_p = m_body * 9.81 * l_COM 
    
    # New parameters for upper and lower body segments
    # Based on standard anthropometric data:
    # Lower limb is ~35% of total height, upper body ~65%
    lower_leg_length = 0.5*H_total  # Lower leg length (~25% of total height)
    upper_body_length = 0.5*H_total  # Upper body length (~50% of total height)
    
    # Mass distribution: lower leg ~10% of body mass, upper body ~60% of body mass
    lower_leg_mass = 0.1 * m_body
    upper_body_mass = 0.6 * m_body
    
    # Center of mass locations
    lower_leg_com = lower_leg_length * 0.45  # CoM at ~45% of lower leg length from ankle
    upper_body_com = upper_body_length * 0.5  # CoM at ~50% of upper body length

    return h_f, m_feet, m_body, l_COM, l_foot, a, K_p, lower_leg_length, upper_body_length, lower_leg_mass, upper_body_mass, lower_leg_com, upper_body_com

def set_geometry_params(root, m_feet, m_body, l_COM, l_foot, a, H_total, h_f, trans_fric, roll_fric, 
                        lower_leg_length=None, upper_body_length=None, lower_leg_mass=None, upper_body_mass=None,
                        lower_leg_com=None, upper_body_com=None):
    
    # If new parameters weren't provided, set defaults
    if lower_leg_length is None:
        lower_leg_length = 0.25*H_total
    if upper_body_length is None:
        upper_body_length = 0.5*H_total
    if lower_leg_mass is None:
        lower_leg_mass = 0.1 * m_body
    if upper_body_mass is None:
        upper_body_mass = 0.6 * m_body
    if lower_leg_com is None:
        lower_leg_com = lower_leg_length * 0.45
    if upper_body_com is None:
        upper_body_com = upper_body_length * 0.5

    for geom in root.iter('geom'):
        if geom.get('name') == "long_link_geom":
            geom.set('mass', "0")
            geom.set('fromto', f'0 0 {lower_leg_length} 0 0 0')  # Modified to use lower leg length
            
        elif geom.get('name') == "m_body":    
            geom.set('mass', str(lower_leg_mass))  # Modified to use lower leg mass
            geom.set('pos', f"0 0 {lower_leg_com}")  # Modified to use lower leg COM
            
        elif geom.get('name') == "upper_body_geom":
            geom.set('fromto', f'0 0 0 0 0 {upper_body_length}')
            
        elif geom.get('name') == "upper_body_com":
            geom.set('mass', str(upper_body_mass))
            geom.set('pos', f"0 0 {upper_body_com}")
            
        elif geom.get('name') == "foot":
            geom.set('pos', f'0 0 0')

    for body in root.iter('body'):
        if body.get('name') == "foot":
            body.set('pos',  f'0. 0 0')
            body.set('quat', f'0 0 0 1')

        elif body.get('name') == "long_link_body":
            body.set('pos', f'{-l_foot/2+a} 0 {h_f}')
            
        elif body.get('name') == "upper_body":
            body.set('pos', f'0 0 {lower_leg_length}')  # Position at top of lower leg

    for joint in root.iter('joint'):
        if joint.get('name') == "ankle_hinge":
            joint.set("pos", f"0 0 0")
            
        elif joint.get('name') == "hip_hinge":  # New hip joint
            joint.set("pos", f"0 0 0")  # Position at base of upper body

        elif joint.get('name') == "rotation_dof":
            joint.set('pos', f'{-l_foot/2+a} 0 {h_f}')

        elif joint.get('name') == "joint_slide_x":
            joint.set('pos', f"{-l_foot/2+a} 0 0.035")

        elif joint.get('name') == "joint_slide_z":
            joint.set('pos', f"{-l_foot/2+a} 0 0.035")

    # Rest of the function remains the same
    for pair in root.iter('pair'):
        if pair.get('name') == "foot_ground_friction":
            pair.set('friction', f"{trans_fric} {trans_fric} 0.99 {roll_fric} {roll_fric}")

    for mesh in root.iter('mesh'):
        if mesh.get('name') == "foot_mesh":
            mesh.set('vertex', f"{-l_foot/2} -0.045 0   {-l_foot/2} 0.045 0   {l_foot/2} -0.045 0   {l_foot/2} 0.045 0  {-l_foot/2+a} -0.045 {h_f} {-l_foot/2+a} 0.045 {h_f}")

    for site in root.iter('site'):
        if site.get('name') == "front_foot_site":
            site.set('fromto', f"{-l_foot/2} 0 0.0 {-l_foot/2} 0 0.1")

        elif site.get('name') == "back_foot_site":
            site.set('fromto', f"{l_foot/2} 0 0.0 {l_foot/2} 0 0.1")

# def set_geometry_params(root, m_feet, m_body, l_COM, l_foot, a, H_total, h_f, trans_fric, roll_fric):

#     for geom in root.iter('geom'):
#         if geom.get('name') == "long_link_geom":
#             geom.set('mass', "0")
#             geom.set('fromto', f'0 0 {H_total} 0 0 0') # this from (x,y,z)_1 to (x,y,z)_2 is w.r.t the long_link_body frame
#             # geom.set('pos', f'0 0 {H_total-l_COM}')

#         elif geom.get('name') == "m_body":    
#             geom.set('mass', str(m_body))
#             geom.set('pos', f"0 0 {l_COM}") # this x,y,z position is w.r.t the long_link_body frame
#             # geom.set('size', 0.05)
        
#         elif geom.get('name') == "foot":
#             geom.set('pos', f'0 0 0') # this x,y,z position is w.r.t the foot_body frame

#     for body in root.iter('body'):
#             if body.get('name') == "foot":
#                 # poo = body.get('pos')
#                 # print(f'pos: {poo}')
#                 body.set('pos',  f'0. 0 0') # this x,y,z position is w.r.t the global frame
#                 body.set('quat', f'0 0 0 1') # unit quaternion for pi radians rotation about global z-axis


#             elif body.get('name') == "long_link_body":
#                 # size = float(body.get('size'))
#                 body.set('pos', f'{-l_foot/2+a} 0 {h_f}') # this x,y,z position is w.r.t the foot reference frame

#     for joint in root.iter('joint'):
#             if joint.get('name') == "ankle_hinge":
#                 joint.set("pos", f"0 0 0") # this x,y,z position is w.r.t the long_link_body reference frame

#             elif joint.get('name') == "rotation_dof":
#                 joint.set('pos', f'{-l_foot/2+a} 0 {h_f}') # this x,y,z position is w.r.t the foot_body reference frame (aligns with world frame)

#             elif joint.get('name') == "joint_slide_x":
#                 joint.set('pos', f"{-l_foot/2+a} 0 0.035") # this x,y,z position is w.r.t the foot_body reference frame (aligns with world frame)

#             elif joint.get('name') == "joint_slide_z":
#                 joint.set('pos', f"{-l_foot/2+a} 0 0.035") # this x,y,z position is w.r.t the foot_body reference frame (aligns with world frame)

#     for pair in root.iter('pair'):
#         if pair.get('name') == "foot_ground_friction":
#             pair.set('friction', f"{trans_fric} {trans_fric} 0.99 {roll_fric} {roll_fric}")

#     for mesh in root.iter('mesh'):
#         if mesh.get('name') == "foot_mesh":
#             mesh.set('vertex', f"{-l_foot/2} -0.045 0   {-l_foot/2} 0.045 0   {l_foot/2} -0.045 0   {l_foot/2} 0.045 0  {-l_foot/2+a} -0.045 {h_f} {-l_foot/2+a} 0.045 {h_f}")

#     for site in root.iter('site'):
#         if site.get('name') == "front_foot_site":
#             # site.set('pos', f"{-l_foot/2} 0 0.05")
#             site.set('fromto', f"{-l_foot/2} 0 0.0 {-l_foot/2} 0 0.1")
#             # site.set('fromto', f"0 0 0.0 0 0 0.1")

#         elif site.get('name') == "back_foot_site":
#             # site.set('pos', f"{l_foot/2} 0 0.05")
#             site.set('fromto', f"{l_foot/2} 0 0.0 {l_foot/2} 0 0.1")
    # for opt in root.iter('option'):
    #     if opt.get('name') == "gravity":
    #         opt.set('')
            # geom.set('mass', str(m_feet))
            # mesh.set('vertex', f"{-l_foot/2} 0 0  {l_foot/2} 0 0  0 -0.035 0  0 0.035 0  {l_foot/2-a} 0 {h_f}")
