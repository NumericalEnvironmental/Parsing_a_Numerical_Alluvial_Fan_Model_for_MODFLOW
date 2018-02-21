#################################################################################
#
# FanGrid.py
#
# python script to develop MODFLOW and PHAST grids from a simulated alluvial fan
#
# by Walt McNab
#
#################################################################################

from __future__ import print_function
from numpy import *
from pandas import *
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

options.mode.chained_assignment = None

class Sediment:
    
    def __init__(self):
        # read sediment properties
        self.name = []
        self.K = []
        lineInput = []        
        inputFile = open('sed_props.txt','r')
        for line in inputFile: lineInput.append(line.split())
        inputFile.close()
        for i in xrange(1, len(lineInput)):
            self.name.append(lineInput[i][0])
            self.K.append(float(lineInput[i][1]))
        self.K = array(self.K)
        print('Read sediment properties.')

    def CalcK(self, fractions):
        # sediment size class weighted hydraulic conductivity
        K = dot(self.K, fractions)
        return 10**K


class Params:
    
    def __init__(self):
         # grid parameters
        lineInput = []        
        inputFile = open('model_params.txt','r')
        for line in inputFile: lineInput.append(line.split())
        inputFile.close()
        self.numLayers = int(lineInput[0][1])               # collapse to this number of layers
        self.thickMin = float(lineInput[1][1])              # minimum thickness for active cells
        self.zMax = float(lineInput[2][1])                  # maximum elevation (anything above is truncated)
        self.dz = float(lineInput[3][1])                    # elevation step size (same as in alluvial fan model)
        self.numBins = int(lineInput[4][1])                 # number of bins for mineralogical class (for PHAST model)
        print('Read model grid constraints.')


class Fan:

    def __init__(self, grid, sed):   
        # import output of alluvial fan generator and convert to data frame
        self.fan_df = read_csv('sed_distribution.csv', sep=',')
        self.fan_df = self.fan_df[self.fan_df['z']<=grid.zMax]
        self.xSet = list(set(self.fan_df['x']))
        self.ySet = list(set(self.fan_df['y']))
        self.xSet.sort()
        self.ySet.sort()
        self.numRows = len(self.ySet)
        logK = dot(self.fan_df[sed.name], log10(sed.K))
        self.fan_df['K'] = 10.**logK
        self.epsilon = 0.01                      # factor used to correct layer numbering scheme
        print('Read fan model output.')               

    def Layers(self, grid):
        # return data frames based on fan_df, but vertically aggregated into grid.numLayers
        print('Processing ...')
        aggStatus = False
        for i, xCol in enumerate(self.xSet):
            for j, yRow in enumerate(self.ySet):
                column_df = self.fan_df[(self.fan_df['x']==xCol) & (self.fan_df['y']==yRow)]
                if len(column_df) < grid.numLayers+1:
                    # add extra sediment slabs to this location
                    for k in xrange(len(column_df), grid.numLayers+1):
                        column_df = column_df.append(column_df.tail(1), ignore_index=True)
                        column_df['z'][k] = column_df['z'][k-1] + grid.dz
                z0 = column_df['z'].min()
                zExtent = column_df['z'].max() - z0 + self.epsilon
                dz = zExtent/grid.numLayers 
                column_df['layer'] = (column_df['z']-z0)/dz
                column_df['layer'] = column_df['layer'].astype('int64')       
                column_df = column_df.groupby('layer').mean()   # collapse into grid.numLayers
                column_df.reset_index(inplace=True)         
                column_df['row'] = self.numRows-j 		# renumber to match MODFLOW row numbering scheme
                column_df['col'] = i+1
                column_df['zBase'] = z0 + column_df['layer'] * dz
                column_df['zMid'] = column_df['zBase'] + 0.5*dz                       
                column_df['zTop'] = column_df['zBase'] + dz                         
                column_df['ibound'] = bool(zExtent>=grid.thickMin) 	# fan areas that are too thin are marked as inactive
                column_df = column_df[['x', 'y', 'zMid', 'col', 'row', 'layer', 'zBase', 'zTop', 'K', 'ibound']] 
                if not aggStatus:
                    layers_df = column_df.copy()
                    aggStatus = True
                else:
                    layers_df = concat([layers_df, column_df], axis=0)
        layers_df.reset_index(inplace=True)
        
        # hydrology data are 3-dimensional
        layers_df['layer'] = grid.numLayers - layers_df['layer']            # MODFLOW layer numbering is top-down
        hydro_df = layers_df[['col', 'row', 'layer', 'K', 'ibound']]        # for MODFLOW
        Kfield_df = layers_df[['x', 'y', 'zMid', 'K']]                      # K-field grid, for interpolation in PHAST
        print('Generated hydrology data sets.')

        # structural model entails only one (x ,y) slice; set up
        layerStruct_df = layers_df[layers_df['layer']==1]   
        zb = array(layerStruct_df['zBase'])
        topModel = array(layerStruct_df['zTop'])
        topSurface_df = layerStruct_df[['x', 'y']]          # create top surface elevation data frame, just for PHAST
        topSurface_df['surface'] = topModel

        # create perimeter delineation set for active cells (for PHAST)
        points_df = layerStruct_df[layerStruct_df['ibound']==True]
        points = array(points_df[['x', 'y']])
        hull = ConvexHull(points)                      
        perimPoints = transpose(points[hull.vertices])                     
        perimPoints_df = DataFrame(data={'x': perimPoints[0], 'y': perimPoints[1]})             
         
        # format and populate layer elevation file for MODFLOW   
        layerStruct_df = layerStruct_df[['col', 'row']]
        layerStruct_df['top_model'] = topModel
        layerStruct_df['base_1'] = zb
        for i in xrange(2, grid.numLayers+1):
            nextLayer_df = layers_df[layers_df['layer']==i]
            zb = array(nextLayer_df['zBase'])
            layerStruct_df['base_' + str(i)] = zb
        print('Generated layer structure data sets.')
        return hydro_df, Kfield_df, layerStruct_df, topSurface_df, perimPoints_df


def Bin(A,n):
    # divide array A into bins by equal-interval method; return bin index numbers
    bins = linspace(min(A), max(A), n, endpoint=False)
    return digitize(A, bins)

### main code ###

def FanGrid():

    # set up sediment class
    sed = Sediment()    
    
    # read grid parameter constraints
    grid = Params()
    
    # import and process prior fan model output
    fan = Fan(grid, sed)
    
    # collapse vertical sediment stacks into layers
    hydro_df, Kfield_df, layerStruct_df, topSurface_df, perimPoints_df = fan.Layers(grid)
    active_df = hydro_df[hydro_df['ibound']==1]
    logK = log10(active_df['K'])		# plot log K histogram (over all active cells) 
    plt.figure()
    ax = logK.plot.hist(bins=40)
    ax.set_xlabel('Log K')
    plt.show()

    # digitize mineralogy around (log) K
    logK = array(log10(Kfield_df['K']))
    minBins = Bin(logK, grid.numBins)
    minField_df = Kfield_df.copy()
    minField_df['bins'] = minBins-1         # adjust indexing to start from 0
    minField_df = minField_df[['x', 'y', 'zMid', 'bins']] 

    print('Writing output.')    
    hydro_df.to_csv('hydro.csv', index=False, sep='\t')
    Kfield_df.to_csv('K_field.csv', index=False, sep='\t')    
    layerStruct_df.to_csv('layers_struct.csv', index=False, sep='\t')
    topSurface_df.to_csv('top_surface.csv', index=False, sep='\t')
    minField_df.to_csv('min_field.csv', index=False, sep='\t')
    perimPoints_df.to_csv('perim_points.csv', index=False, sep='\t')

    print('Done.')

    
### run script ###

FanGrid()