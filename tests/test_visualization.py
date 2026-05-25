import matplotlib

matplotlib.use("Agg")


def test_style_preset_dpi_is_not_rc_param():
    import matplotlib.pyplot as plt

    from oneehr.visualization._style import new_figure

    fig, ax = new_figure(style="nature")

    assert ax.figure is fig
    assert fig.dpi == 300
    plt.close(fig)
