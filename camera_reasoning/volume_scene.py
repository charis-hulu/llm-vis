import numpy as np
import vtk


def load_raw_volume(path: str, dimensions: tuple, scalar_type: str = "uint8") -> vtk.vtkImageData:
    """Read a raw binary volume file into a vtkImageData object."""
    dx, dy, dz = dimensions
    dtype = np.dtype(scalar_type)
    data = np.fromfile(path, dtype=dtype)
    expected = dx * dy * dz
    if data.size != expected:
        raise ValueError(
            f"Raw file has {data.size} values but dimensions {dimensions} require {expected}."
        )
    # VTK expects Fortran-order (x varies fastest) — raw volumes are typically C-order
    data = data.reshape((dz, dy, dx))

    image = vtk.vtkImageData()
    image.SetDimensions(dx, dy, dz)
    image.SetOrigin(0.0, 0.0, 0.0)
    image.SetSpacing(1.0, 1.0, 1.0)

    vtk_array = vtk.util.numpy_support.numpy_to_vtk(
        data.ravel(order="C"), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
    )
    image.GetPointData().SetScalars(vtk_array)
    return image


def build_isosurface_pipeline(image: vtk.vtkImageData, isovalue: float):
    """Return (actor, renderer, render_window) for an isosurface render."""
    mc = vtk.vtkFlyingEdges3D()
    mc.SetInputData(image)
    mc.SetValue(0, isovalue)
    mc.ComputeNormalsOn()
    mc.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(mc.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.85, 0.75, 0.65)
    actor.GetProperty().SetSpecular(0.3)
    actor.GetProperty().SetSpecularPower(20)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.1, 0.1)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 800)
    render_window.SetOffScreenRendering(1)

    return actor, renderer, render_window


def save_screenshot(render_window: vtk.vtkRenderWindow, path: str):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    render_window.Render()
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(render_window)
    w2i.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(path))
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()
