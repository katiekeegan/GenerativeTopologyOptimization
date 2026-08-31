def test_topology_format_converter_dependency_imports():
    import topology_format_converter

    assert hasattr(topology_format_converter, "density_to_training_cache")
    assert hasattr(topology_format_converter, "repair_mesh")
    assert hasattr(topology_format_converter, "save_volume_vtk")
