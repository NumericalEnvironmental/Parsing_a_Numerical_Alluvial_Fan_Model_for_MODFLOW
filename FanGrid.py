############################################################################
#
# FanGrid.py
#
# python script to develop a MODFLOW grid from a simulated alluvial fan
#
# by Walt McNab
#
############################################################################

from __future__ import print_function
from numpy import *
from pandas import *
import matplotlib.pyplot as plt

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
        self.numLayers = int(lineInput[0][1])                # grid origin
        self.thickMin = float(lineInput[1][1])
        self.zMax = float(lineInput[2][1])
        self.dz = float(lineInput[3][1])
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
        # return dataframes based on fan_df, but vertically aggregated into grid.numLayers
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
                column_df = column_df.groupby('layer').mean()
                column_df.reset_index(inplace=True)         
                column_df['row'] = self.numRows-j 		# renumber to match MODFLOW row numbering scheme
                column_df['col'] = i+1
                column_df['zBase'] = z0 + column_df['layer']*dz                         
                column_df['zTop'] = column_df['zBase'] + dz                         
                column_df['ibound'] = bool(zExtent>=grid.thickMin) 	# fan areas that are too thin are marked as inactive
                column_df = column_df[['col', 'row', 'layer', 'zBase', 'zTop', 'K', 'ibound']]     
                if not aggStatus:
                    layers_df = column_df.copy()
                    aggStatus = True
                else: layers_df = concat([layers_df, column_df], axis=0)
        layers_df.reset_index(inplace=True)
        layers_df['layer'] = grid.numLayers - layers_df['layer']
        hydro_df = layers_df[['col', 'row', 'layer', 'K', 'ibound']]        # hydrology props in 3-D
        print('Generated hydrology data set.')        
        layerStruct_df = layers_df[layers_df['layer']==1]
        zb = array(layerStruct_df['zBase'])
        topModel = array(layerStruct_df['zTop'])
        layerStruct_df = layerStruct_df[['col', 'row']] 
        layerStruct_df['top_model'] = topModel
        layerStruct_df['base_1'] = zb
        for i in xrange(2, grid.numLayers+1):
            nextLayer_df = layers_df[layers_df['layer']==i]
            zb = array(nextLayer_df['zBase'])
            layerStruct_df['base_' + str(i)] = zb
        print('Generated layer structure data set.')
        return hydro_df, layerStruct_df


### main code ###

def FanGrid():

    # set up sediment class
    sed = Sediment()    
    
    # read grid parameter constraints
    grid = Params()
    
    # import and process prior fan model output
    fan = Fan(grid, sed)
    
    # collapse vertical sediment stacks into layers
    hydro_df, layerStruct_df = fan.Layers(grid)
    active_df = hydro_df[hydro_df['ibound']==1]
    logK = log10(active_df['K'])		# plot log K histogram (over all active cells) 
    plt.figure()
    ax = logK.plot.hist(bins=40)
    ax.set_xlabel('Log K')
    plt.show()

    print('Writing output.')    
    hydro_df.to_csv('hydro.csv', index=False, sep='\t')
    layerStruct_df.to_csv('layers_struct.csv', index=False, sep='\t')

    print('Done.')

    
### run script ###

FanGrid()