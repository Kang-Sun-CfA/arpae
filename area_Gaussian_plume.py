import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path
from scipy.special import erf

class checkerboard_polygons():
    '''
    construct checkerboard pattern of square emission sources, representing corn fields in the midwest
    '''
    def __init__(self,field_center_x,field_center_y,field_width):
        '''
        field_center_x:
            x coordinates of the centers of squares
        field_center_y:
            y coordinates of the centers of squares
        field_width:
            width of squares. All units should be km
        '''
        field_center_xmesh,field_center_ymesh = np.meshgrid(field_center_x,field_center_y)
        self.field_center_xs = field_center_xmesh.ravel()
        self.field_center_ys = field_center_ymesh.ravel()
        self.field_width = field_width
    
    def evaluate_gplume(self,field_fluxes,xmesh,ymesh,zmesh,wind_speed,wind_direction,
                        plume_class='D',polygon_resolving_unit=None,save_jacobian=False):
        '''
        calling gplume function for each square source. optionally save the jacobian,
        i.e., [d(concentration at receptors)/d(flux at source) for source in all sources]
        search "def gplume" in this code for details
        '''
        xmesh = np.atleast_1d(xmesh)
        ymesh = np.atleast_1d(ymesh)
        zmesh = np.atleast_1d(zmesh)
        self.field_fluxes = field_fluxes[0:len(self.field_center_xs)]
        field_width = self.field_width
        C = np.zeros(xmesh.shape)
        jacobian = []
        for (x,y,Q) in zip(self.field_center_xs,self.field_center_ys,self.field_fluxes):
            xvertices = np.array([x-field_width/2,
                                  x-field_width/2,
                                  x+field_width/2,
                                  x+field_width/2])
            yvertices = np.array([y-field_width/2,
                                  y+field_width/2,
                                  y+field_width/2,
                                  y-field_width/2])
            single_polygon_C,single_polygon_dCdQ = gplume(Q,xmesh,ymesh,zmesh,xvertices,yvertices,wind_speed,wind_direction,
                                        plume_class,polygon_resolving_unit)
            if save_jacobian:
                jacobian.append(single_polygon_dCdQ)
            C += single_polygon_C
        return C.squeeze(),np.array(jacobian).squeeze()
    
    def plot_polygons(self,ax=None,**kwargs):
        '''
        plot the checkerboard pattern
        '''
        if kwargs is None:
            kwargs = {}
        field_width = self.field_width
        if ax is None:
            fig,ax = plt.subplots(1,1,figsize=(10,5))
        for i,(x,y) in enumerate(zip(self.field_center_xs,self.field_center_ys)):
            xvertices = np.array([x-field_width/2,
                                  x-field_width/2,
                                  x+field_width/2,
                                  x+field_width/2])
            yvertices = np.array([y-field_width/2,
                                  y+field_width/2,
                                  y+field_width/2,
                                  y-field_width/2])
            ax.plot(np.append(xvertices,xvertices[0]),np.append(yvertices,yvertices[0]),**kwargs)
            ax.text(x,y,'source {}'.format(i+1),va='center',ha='center')
        ax.set_aspect('equal')

def gplume(Q,xmesh,ymesh,zmesh,xvertices,yvertices,wind_speed,wind_direction,
           plume_class='D',polygon_resolving_unit=None):
    '''
    Gaussian plume model from an area emission source. Following Smith (1993)
    Dispersion of Odours From Ground Level Agricultural Sources.
    Q:
        area-average flux in mass unit/m2/s. It's Qa in Smith 1993
    x/y/zmesh:
        3D coordinates to evaluate the model. unit should be km for xmesh and ymesh, but m for zmesh
        units will be converted to m in the function
    x/yvertices:
        corner points of the source polygon. unit should be km
        units will be converted to m in the function
    wind_speed:
        mean wind speed. m/s
    wind_direction:
        0/360: north; 90: east; 270: west; 180: south
    plume_class:
        Pasquill stability class. See, for example, https://www.ready.noaa.gov/READYpgclass.php
    polygon_resolving_unit:
        level of discretization for the polygon. by default, discretize the smaller one in the vertical/horizontal
        extents by a factor of 50
    returns:
        C: concentration in mass unit/m3
        dCdQ: derivative of concentration to flux
    '''
    xmesh = np.atleast_1d(xmesh)
    ymesh = np.atleast_1d(ymesh)
    zmesh = np.atleast_1d(zmesh)
    angle = -(270-wind_direction)/360*2*np.pi;
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle),  np.cos(angle)]])
    # rotate vertices to align wind direction, convert km to m
    xy1 = np.array([xvertices,yvertices])*1000
    xy2 = rotation_matrix@xy1

    xv = xy2[0,:]
    yv = xy2[1,:]

    shiftx = np.mean(xv)
    shifty = np.mean(yv)
    # rotate reference frame to align wind direction, convert km to m
    xym1 = np.array([xmesh.ravel(),ymesh.ravel()])*1000
    xym2 = rotation_matrix@xym1

    xinput = xym2[0,:].reshape(xmesh.shape)-shiftx
    yinput = xym2[1,:].reshape(xmesh.shape)-shifty
    zinput = zmesh

    xv -= shiftx
    yv -= shifty

    polygon = path.Path([(x,y) for (x,y) in zip(xv,yv)])
    # discretize the polygon
    if polygon_resolving_unit is None:
        polygon_resolving_unit = np.min([np.ptp(xv),np.ptp(yv)])/50
    xstep = np.linspace(np.min(xv),np.max(xv),int(np.ceil(np.ptp(xv)/polygon_resolving_unit)))
    ystep = np.linspace(np.min(yv),np.max(yv),int(np.ceil(np.ptp(yv)/polygon_resolving_unit)))
    xstepmesh,ystepmesh = np.meshgrid(xstep,ystep)

    all_points = np.hstack((xstepmesh.ravel()[:,np.newaxis],ystepmesh.ravel()[:,np.newaxis]))
    in_mask = polygon.contains_points(all_points).reshape(xstepmesh.shape)
    # length of strip
    Yi = np.sum(in_mask,0)/len(ystep)*np.ptp(yv)
    Yi1 = np.zeros_like(Yi)
    yi = np.zeros_like(Yi)
    xi = xstep.copy()
    deltax = np.abs(np.median(np.diff(xstep)))

    ay = 0.34;  by = 0.82;  az = 0.275; bz = 0.82;  
    sumterm = 0
    np.seterr(divide='ignore',invalid='ignore',over='ignore'); #turn off warning
    for i in range(len(xstep)):
        in_idx = np.nonzero(in_mask[:,i])[0]
        if len(in_idx) == 0:
            continue
        high = ystep[in_idx[-1]]
        low = ystep[in_idx[0]]
        Yi1[i] = np.abs(high-low)
        yi[i] = np.mean([high,low])

        xtemp = xinput-xi[i]
        if plume_class == 'G':
            sigmayi = ay*np.power(np.abs(xtemp),by)
            sigmayi[xtemp<=0] = 0
            sigmazi = az*np.power(np.abs(xtemp),bz)
            sigmazi[xtemp<=0] = 0
        else:
            sigmayi,sigmazi = gsigma(xtemp,plume_class)
            sigmayi[xtemp<=0] = 0
            sigmazi[xtemp<=0] = 0
        if Yi[i] == 0:
            addterm = 0
        else:
            addterm = 1/np.sqrt(2*np.pi)/sigmazi*np.exp(-np.power(zinput,2)/2/np.power(sigmazi,2))\
            *(erf((yinput-yi[i]+Yi[i]/2)/np.sqrt(2*np.power(sigmayi,2)))\
              -erf((yinput-yi[i]-Yi[i]/2)/np.sqrt(2*np.power(sigmayi,2))))
            addterm[np.isnan(addterm)] = 0
            addterm[np.isinf(addterm)] = 0
        sumterm += addterm
    sumterm[np.isnan(sumterm)] = 0
    sumterm[np.isinf(sumterm)] = 0
    C = sumterm*deltax*Q/wind_speed
    dCdQ = sumterm*deltax/wind_speed
    np.seterr(divide='warn',invalid='warn',over='warn'); #turn back on warning
    return C,dCdQ

def gsigma(x,plume_class):
    '''
    cross wind (a and b) and vertical dispersion parameters (c and d) from Pasquill
    stability class. Numbers are from Tables 1 and 2 of Smith 1993
    x:
        downwind distance, unit should be m
    plume_class:
        letter indicating the Pasquill stability class
    '''
    c = np.zeros_like(x)
    d = np.zeros_like(x)
    if plume_class == 'A':
        a = 0.72722;b = 0.044216;
        c[x <= 150] = 0.1087;d[x <= 150] = 1.0542;
        c[(x > 150) & (x <= 200)] = 0.08942; d[(x > 150) & (x <= 200)] = 1.0932;
        c[(x > 200) & (x <= 250)] = 0.07058; d[(x > 200) & (x <= 250)] = 1.1262;
        c[(x > 250) & (x <= 300)] = 0.035; d[(x > 250) & (x <= 300)] = 1.2644;
        c[(x > 300) & (x <= 400)] = 0.01531; d[(x > 300) & (x <= 400)] = 1.4094;
        c[(x > 400) & (x <= 500)] = 0.002265; d[(x > 400) & (x <= 500)] = 1.7283;
        c[x > 500] = 0.0002028; d[x > 500] = 2.1166;
    elif plume_class == 'B':
        a = 0.53814; b = 0.031583;
        c[x <= 200]= 0.1451; d[x <= 200] = 0.93198;
        c[(x > 200) & (x <= 400)] = 0.1105; d[(x > 200) & (x <= 400)] = 0.98332;
        c[x > 400] = 0.05589; d[x > 400] = 1.0971;
    elif plume_class == 'C':
        a = 0.34906; b = 0.018949;
        c = 0.1103; d = 0.91465;
    elif plume_class == 'D':
        a = 0.2327; b = 0.012633;
        c[(x > 0) & (x <= 300)] = 0.08474; d[(x > 0) & (x <= 300)] = 0.86974;
        c[(x > 300) & (x <= 1000)] = 0.1187; d[(x > 300) & (x <=1000)] = 0.81066;
        c[(x > 1000) & (x <= 3000)] = 0.3752; d[(x > 1000) & (x <= 3000)] = 0.64403;
        c[(x > 3000) & (x <= 10000)] = 0.5125; d[(x > 3000) & (x <= 10000)] = 0.60486;
    elif plume_class == 'E':
        a = 0.17453; b = 0.009475;
        c[(x > 0) & (x <= 300)] = 0.08144; d[(x > 0) & (x <= 300)] = 0.81956;
        c[(x > 300) & (x <= 1000)] = 0.1162; d[(x > 300) & (x <=1000)] = 0.7566;
        c[(x > 1000) & (x <= 2000)] = 0.2771; d[(x > 1000) & (x <= 2000)] = 0.63077;
        c[(x > 2000) & (x <= 4000)] = 0.4347; d[(x > 2000) & (x <= 4000)] = 0.57144;
        c[(x > 4000) & (x <= 10000)] = 0.7533; d[(x > 4000) & (x <= 10000)] = 0.50527;
    elif plume_class == 'F':
        a = 0.11636; b = 0.006317;
        c[(x > 0) & (x <= 200)] = 0.05437; d[(x > 0) & (x <= 200)] = 0.81588;
        c[(x > 200) & (x <= 700)] = 0.06425; d[(x > 200) & (x <=700)] = 0.78407;
        c[(x > 700) & (x <= 1000)] = 0.1232; d[(x > 700) & (x <= 1000)] = 0.68465;
        c[(x > 1000) & (x <= 2000)] = 0.177; d[(x > 1000) & (x <= 2000)] = 0.63227;
        c[(x > 2000) & (x <= 3000)] = 0.3434; d[(x > 2000) & (x <= 3000)] = 0.54503;
        c[(x > 3000) & (x <= 7000)] = 0.6523; d[(x > 3000) & (x <= 7000)] = 0.4649;

    sigmayi = 0.84678*x*np.tan(a-b*np.log(x))
    sigmazi = c*np.power(x,d)
    return sigmayi,sigmazi