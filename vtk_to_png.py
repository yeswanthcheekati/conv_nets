# import the simple module from the paraview
from paraview.simple import *
# disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

for i in range(225):
    file_name = '/Users/yeswanthcheekati/Downloads/SU2-5.0.0/samp/airfoil/data/sample_' + \
        str(i)+'/flow.vtk'
    # create a new 'Legacy VTK Reader'
    flowvtk = LegacyVTKReader(FileNames=[file_name])

    # set active source
    SetActiveSource(flowvtk)

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')
    # uncomment following to set a specific view size
    # renderView1.ViewSize = [1048, 924]

    # get color transfer function/color map for 'Conservative_1'
    conservative_1LUT = GetColorTransferFunction('Conservative_1')
    conservative_1LUT.RGBPoints = [1.2011480066576041e-05, 0.231373, 0.298039, 0.752941, 1.2789304946636548e-05,
                                   0.865003, 0.865003, 0.865003, 1.3567129826697055e-05, 0.705882, 0.0156863, 0.14902]
    conservative_1LUT.ScalarRangeInitialized = 1.0

    # get opacity transfer function/opacity map for 'Conservative_1'
    conservative_1PWF = GetOpacityTransferFunction('Conservative_1')
    conservative_1PWF.Points = [1.2011480066576041e-05,
                                0.0, 0.5, 0.0, 1.3567129826697055e-05, 1.0, 0.5, 0.0]
    conservative_1PWF.ScalarRangeInitialized = 1

    # show data in view
    flowvtkDisplay = Show(flowvtk, renderView1)
    # trace defaults for the display properties.
    flowvtkDisplay.Representation = 'Surface'
    flowvtkDisplay.ColorArrayName = ['POINTS', 'Conservative_1']
    flowvtkDisplay.LookupTable = conservative_1LUT
    flowvtkDisplay.OSPRayScaleArray = 'Conservative_1'
    flowvtkDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    flowvtkDisplay.SelectOrientationVectors = 'Conservative_1'
    flowvtkDisplay.ScaleFactor = 2.0
    flowvtkDisplay.SelectScaleArray = 'Conservative_1'
    flowvtkDisplay.GlyphType = 'Arrow'
    flowvtkDisplay.GlyphTableIndexArray = 'Conservative_1'
    flowvtkDisplay.DataAxesGrid = 'GridAxesRepresentation'
    flowvtkDisplay.PolarAxes = 'PolarAxesRepresentation'
    flowvtkDisplay.ScalarOpacityFunction = conservative_1PWF
    flowvtkDisplay.ScalarOpacityUnitDistance = 0.8573928093340814
    flowvtkDisplay.GaussianRadius = 1.0
    flowvtkDisplay.SetScaleArray = ['POINTS', 'Conservative_1']
    flowvtkDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    flowvtkDisplay.OpacityArray = ['POINTS', 'Conservative_1']
    flowvtkDisplay.OpacityTransferFunction = 'PiecewiseFunction'

    # show color bar/color legend
    flowvtkDisplay.SetScalarBarVisibility(renderView1, True)

    # reset view to fit data
    renderView1.ResetCamera()

    # set scalar coloring
    ColorBy(flowvtkDisplay, ('POINTS', 'Mach'))

    # Hide the scalar bar for this color map if no visible data is colored by
    # it.
    HideScalarBarIfNotNeeded(conservative_1LUT, renderView1)

    # rescale color and/or opacity maps used to include current data range
    flowvtkDisplay.RescaleTransferFunctionToDataRange(True, False)

    # show color bar/color legend
    flowvtkDisplay.SetScalarBarVisibility(renderView1, True)

    # get color transfer function/color map for 'Mach'
    machLUT = GetColorTransferFunction('Mach')
    machLUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941, 0.11076229810714722,
                         0.865003, 0.865003, 0.865003, 0.22152459621429443, 0.705882, 0.0156863, 0.14902]
    machLUT.ScalarRangeInitialized = 1.0

    # Apply a preset using its name. Note this may not work as expected when
    # presets have duplicate names.
    machLUT.ApplyPreset('Rainbow Desaturated', True)

    # Rescale transfer function
    machLUT.RescaleTransferFunction(0.0, 0.2)

    # get opacity transfer function/opacity map for 'Mach'
    machPWF = GetOpacityTransferFunction('Mach')
    machPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.22152459621429443, 1.0, 0.5, 0.0]
    machPWF.ScalarRangeInitialized = 1

    # Rescale transfer function
    machPWF.RescaleTransferFunction(0.0, 0.2)

    # Properties modified on renderView1
    renderView1.OrientationAxesVisibility = 0

    # hide color bar/color legend
    flowvtkDisplay.SetScalarBarVisibility(renderView1, False)

    # current camera placement for renderView1
    renderView1.CameraPosition = [5.0, 0.0, 47.162509277084176]
    renderView1.CameraFocalPoint = [5.0, 0.0, 0.0]
    renderView1.CameraParallelScale = 12.206555615733702
    renderView1.ResetCamera(-15.0, 15.0, -15.0, 15.0, 0.0, 0.0)

    # save screenshot
    SaveScreenshot('/Users/yeswanthcheekati/Desktop/data/foil/label_' +
                   str(i)+'.png', renderView1, ImageResolution=[4192, 3696])

    renderView1.ResetCamera(-15.0, 15.0, -15.0, 15.0, 0.0, 0.0)

