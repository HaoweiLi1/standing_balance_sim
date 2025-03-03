import xml.etree.ElementTree as ET

# calculate mass, geometry, and controller gain data from literature equations
def calculate_kp_and_geom(weight, height):
    
    M_total = weight # [kg]
    H_total = height # [meters]

    m_feet = 2*0.0145 * M_total
    m_legs = 2 * 0.0465 * M_total  # Both legs (4.65% each)
    m_torso = M_total - (m_feet + m_legs)  # Rest is torso, arms, head

    l_foot = 0.152 * H_total
    l_leg = 0.246 * H_total  # Leg length (including thigh and shank)
    l_torso = 0.5 * H_total  # Upper body length

    h_f = 0.039*H_total
    a = 0.19*l_foot

    # CoM positions
    l_COM_leg = 0.447 * l_leg  # CoM position along leg (from ankle joint)
    l_COM_torso = 0.5 * l_torso  # CoM position along torso (from hip joint)
   

    return h_f, m_feet, m_legs, m_torso, l_leg, l_torso, l_COM_leg, l_COM_torso, l_foot, a


def set_geometry_params(root, m_feet, m_legs, m_torso, l_leg, l_torso, l_COM_leg, l_COM_torso, l_foot, a, H_total, h_f, trans_fric, roll_fric):

    for geom in root.iter('geom'):
        # Update leg geometry
        if geom.get('name') == "leg_link_geom":
            geom.set('mass', str(m_legs))
            geom.set('fromto', f'0 0 0 0 0 {l_leg}')
            
        # Update torso geometry
        elif geom.get('name') == "torso_link_geom":
            geom.set('mass', "0")  # Set mass to 0 since we'll use a separate CoM
            geom.set('fromto', f'0 0 0 0 0 {l_torso}')
            
        # Update body mass point for torso
        elif geom.get('name') == "m_body":
            geom.set('mass', str(m_torso))
            geom.set('pos', f"0 0 {l_COM_torso}")  # Position at torso's CoM
            
        # Update foot geometry
        elif geom.get('name') == "foot":
            geom.set('pos', f'0 0 0')  # Position relative to foot_body frame

    for body in root.iter('body'):
        # Update foot position
        if body.get('name') == "foot":
            body.set('pos', f'0. 0 0')  # Position in global frame
            body.set('quat', f'0 0 0 1')  # Unit quaternion
            
        # Update leg segment position
        elif body.get('name') == "leg_segment":
            body.set('pos', f'{-l_foot/2+a} 0 {h_f}')  # Position relative to foot frame
            
        # Update torso segment position
        elif body.get('name') == "torso_segment":
            body.set('pos', f'0 0 {l_leg}')  # Position at top of leg

    for joint in root.iter('joint'):
        # Update ankle joint
        if joint.get('name') == "ankle_hinge":
            joint.set("pos", f"0 0 0")  # Position relative to leg segment
            
        # Update hip joint
        elif joint.get('name') == "hip_hinge":
            joint.set("pos", f"0 0 0")  # Position relative to torso segment
            
        # Update other joints
        elif joint.get('name') == "rotation_dof":
            joint.set('pos', f'{-l_foot/2+a} 0 {h_f}')
            
        elif joint.get('name') == "joint_slide_x":
            joint.set('pos', f"{-l_foot/2+a} 0 0.035")
            
        elif joint.get('name') == "joint_slide_z":
            joint.set('pos', f"{-l_foot/2+a} 0 0.035")

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
