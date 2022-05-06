import numpy as np
import math

# Declaring useful constants about the system
rotAxis_x = -0.58       #mm
rotAxis_x_unc = 0.2     #mm
rotAxis_y = -0.77       #mm
rotAxis_y_unc = 0.07    #mm
rotZero_ang = 64.65     #deg
rotZero_ang_unc = 0.15  #deg
rotZero_x = 0           #mm
rotZero_x_unc = 0.25    #mm
rotZero_y = -0.92       #mm
rotZero_y_unc = 0.14    #mm
col_offset = 0.9        #mm
col_offset_unc = 0.07    #mm
linear_center = 2.5     #mm
majorAxis = 217         #mm
minorAxis = 183         #mm
linMotor_unc = 0.01     #mm
rotMotor_unc = 0.07     #deg
sourceMotor_unc = 2.2   #deg
sourceAxToDet = 25.1    #mm, for ICPC
ditchDepth = 2          #mm, for ICPC


toRad = math.pi/180

def main():
    radii = [15]
    thetaRots = [0, 145, 180]
    thetaDets = [90, 75, 60, 45]
    ditchScan = True
    for radius in radii:
        for thetaRot in thetaRots:
            for thetaDet in thetaDets:
                print(f"Rotary angle: {thetaRot}, Radius: {radius}, Detector Angle: {thetaDet}")
                rotary, linear, source = calculateMotorPos(radius, thetaRot, thetaDet)
                print(f"Move rotary motor to: -{rotary} degrees")
                print(f"Move linear motor to: {linear:0.3f} mm")
                print(f"Move source motor to: {source} degrees")
                pos_x, pos_x_unc, pos_y, pos_y_unc = calculateDetPos(rotary, linear, source, ditch=ditchScan)
                print(f"This will put us at:     ({pos_x:0.3f} +- {pos_x_unc:0.3f}, {pos_y:0.3f} +- {pos_y_unc:0.3f})")
                targetX = radius*math.sin(thetaRot*toRad)
                targetY = radius*math.cos(thetaRot*toRad)
                dist = np.sqrt((targetY-pos_y)**2 + (targetX-pos_x)**2)
                dist_unc = np.sqrt( ((pos_x-targetX)**2/dist**2) * pos_x_unc**2 + ((pos_y-targetY)**2/dist**2) * pos_y_unc**2 )
                print(f"This is {dist:0.3f} +- {dist_unc:0.3f} mm away from the target")
                print()

def calculateMotorPos(radius, thetaRot, thetaDet=90):
    '''Gives the motor commands that should be called to get the collimator beam to the given position on the detector
    
    radius: distance from center of detector [mm]
    thetaRot: azimuthal angle where zero aligns with the motor rotary zero position [degrees]
    thetaDet: angle that the beam makes wrt to the detector surface (90 is normal incidence) [degrees] 
    
    Coordinate system: 
    (0,0) is the center of the detector/cold plate
    +x is along the minor axis of the top hat, closer to the diving board
    -x is along the minor axis of the top hat, moving away from the diving board
    +y is along the major axis of the top hat, positive linear motion at rotary 0
    -y is along the major axis of the top hat, negative linear motion at rotary 0
    '''
    targetX = radius*math.sin(thetaRot*toRad)
    targetY = radius*math.cos(thetaRot*toRad)
    print(f"Target: ({targetX}, {targetY})")

    angle, linear, source = calculateMotorFromTarget(targetX, targetY, thetaRot, thetaDet)

    if linear < -linear_center:
        print(angle, linear, source)
        print("We need to turn around")
        return calculateMotorFromTarget(targetX, targetY, angle+180, 180-thetaDet, flipped=True)
        #return calculateMotorPos(-radius, angle+180, 180-thetaDet, flipped=True)
        #return angle+180, -(linear_center+linear), -180-(source+180)

    return angle, linear, source

def calculateMotorFromTarget(targetX, targetY, thetaRot, thetaDet=90, flipped=False):
    radius = np.sqrt(targetX**2 + targetY**2)
    distance = 1e8
    angle = 0
    closestX = 0
    closestY = 0
    centerX = 0
    centerY = 0
    for r in np.arange(thetaRot-25, thetaRot+6):
        rot_rad = r*toRad
        if rot_rad == 0:
            rot_rad = 1e-15
        center_x, _, center_y, _ = rotate(rot_rad) 
        d = np.abs(math.cos((90-r)*toRad)*(center_y - targetY) - math.sin((90-r)*toRad)*(center_x - targetX))
        if d < distance:
            slope = 1/math.tan(rot_rad) 
            intercept = center_y - center_x*slope 
            x = (targetX + slope*targetY - slope*intercept)/(slope**2 + 1)
            y = (slope*(targetX + slope*targetY) + intercept)/(slope**2 + 1)

            distance = d
            angle = r
            closestX = x 
            closestY = y 
            centerX = center_x
            centerY = center_y
    
    sourceLinearDist = 0
    if thetaDet != 90: #only need to care if not normal incidence
        y = sourceAxToDet
        if 13 < np.abs(radius) < 16:
            y += ditchDepth
        sourceLinearDist = y/math.tan(thetaDet*toRad)
    

    linear = np.sqrt((closestY - centerY)**2 + (closestX - centerX)**2) + col_offset - sourceLinearDist 
    if flipped:
        linear = -np.sqrt((closestY - centerY)**2 + (closestX - centerX)**2) + col_offset - sourceLinearDist 
    source = -180 + (90-thetaDet)

    return angle, linear, source
    

def rotate(rot_rad, point_x=rotZero_x, point_x_unc=rotZero_x_unc, point_y=rotZero_y, point_y_unc=rotZero_y_unc, axis_x=rotAxis_x, axis_x_unc=rotAxis_x_unc, axis_y=rotAxis_y, axis_y_unc=rotAxis_y_unc):
    '''
    Rotate a point about an axis through an angle

    rot_rad: the rotation angle, in radians
    (point_x +- point_x_unc, point_y +- point_y_unc): the point to be rotated
    (axis_x +- axis_x_unc, axis_y +- axis_y_unc): the rotation axis
    '''
    if rot_rad == 0:
        rot_rad = 1e-20
    rotX = math.cos(rot_rad)*(point_x - axis_x) + math.sin(rot_rad)*(point_y - axis_y) + axis_x
    rotY = -math.sin(rot_rad)*(point_x - axis_x) + math.cos(rot_rad)*(point_y - axis_y) + axis_y
    xdif_unc = np.sqrt(point_x_unc**2 + axis_x_unc**2)
    ydif_unc = np.sqrt(point_y_unc**2 + axis_y_unc**2)
    cos_unc = math.sin(rot_rad)*rotMotor_unc*toRad 
    sin_unc = math.cos(rot_rad)*rotMotor_unc*toRad

    #TODO: Fix this so it doesn't give divie by zero errors at rot0, for now, use rotary 1e-32 or whatever
    x_unc1 = math.cos(rot_rad)*(point_x - axis_x) * np.sqrt( (cos_unc/math.cos(rot_rad))**2 + (xdif_unc/(point_x - axis_x))**2 )
    x_unc2 = math.sin(rot_rad)*(point_y - axis_y) * np.sqrt( (sin_unc/math.sin(rot_rad))**2 + (ydif_unc/(point_y - axis_y))**2 )
    y_unc1 = math.sin(rot_rad)*(point_x - axis_x) * np.sqrt( (sin_unc/math.sin(rot_rad))**2 + (xdif_unc/(point_x - axis_x))**2 )
    y_unc2 = math.cos(rot_rad)*(point_y - axis_y) * np.sqrt( (cos_unc/math.cos(rot_rad))**2 + (ydif_unc/(point_y - axis_y))**2 )

    rotX_unc = np.sqrt(x_unc1**2 + x_unc2**2 + axis_x_unc**2)
    rotY_unc = np.sqrt(y_unc1**2 + y_unc2**2 + axis_y_unc**2)
    return rotX, rotX_unc, rotY, rotY_unc

def calculateDetPos(rotary, linear, source=-180, ditch=False):
    '''
    Calculate the beam position on the detector for a given set of motor positions
    rotary: rotary angle, in degrees (0 <= rotary <= 270)
    linear: linear distance measured from "center", which is 2.5mm
    source: source angle, in degrees, 180 is normal incidence
    '''
    rot_rad = rotary*toRad
    if rot_rad == 0:
        rot_rad = 1e-15
        
    center_x, center_x_unc, center_y, center_y_unc = rotate(rot_rad, rotZero_x, rotZero_x_unc, rotZero_y, rotZero_y_unc)
    final_x = center_x + (linear-col_offset)*math.sin(rot_rad) 
    final_y = center_y + (linear-col_offset)*math.cos(rot_rad)
    linx2_unc = (linear*math.sin(rot_rad))*np.sqrt( (linMotor_unc/linear)**2  + ((math.cos(rot_rad)*rotMotor_unc*toRad)/math.sin(rot_rad))**2 )
    liny2_unc = (linear*math.cos(rot_rad))*np.sqrt( (linMotor_unc/linear)**2 + ((math.sin(rot_rad)*rotMotor_unc*toRad)/math.cos(rot_rad))**2 )
    colx2_unc = (col_offset*math.sin(rot_rad))*np.sqrt( (col_offset_unc/col_offset)**2  + ((math.cos(rot_rad)*rotMotor_unc*toRad)/math.sin(rot_rad))**2 )
    coly2_unc = (col_offset*math.cos(rot_rad))*np.sqrt( (col_offset_unc/col_offset)**2 + ((math.sin(rot_rad)*rotMotor_unc*toRad)/math.cos(rot_rad))**2 )
    final_x_unc = np.sqrt(center_x_unc**2 + (linx2_unc)**2 + (colx2_unc)**2)
    final_y_unc = np.sqrt(center_y_unc**2 + (liny2_unc)**2 + (coly2_unc)**2)

    if source != -180: #only need to care if not normal incidence
        y = sourceAxToDet
        if ditch:
            y += ditchDepth 
        sourceLinearDist = y/math.tan(-(90+source)*toRad)
        final_x += sourceLinearDist*math.sin(rot_rad) 
        final_y += sourceLinearDist*math.cos(rot_rad) 

        source_unc = y*sourceMotor_unc*toRad/(math.sin((90+source)*toRad)**2)
        sourcex_unc = (sourceLinearDist*math.sin(rot_rad))*np.sqrt( (source_unc/sourceLinearDist)**2  + ((math.cos(rot_rad)*rotMotor_unc*toRad)/math.sin(rot_rad))**2 )
        sourcey_unc = (sourceLinearDist*math.cos(rot_rad))*np.sqrt( (source_unc/sourceLinearDist)**2 + ((math.sin(rot_rad)*rotMotor_unc*toRad)/math.cos(rot_rad))**2 )
        final_x_unc = np.sqrt( final_x_unc**2 + sourcex_unc**2)
        final_y_unc = np.sqrt( final_y_unc**2 + sourcey_unc**2)


    return final_x, final_x_unc, final_y, final_y_unc 

if __name__ == "__main__":
    main()
