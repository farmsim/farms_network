import time
from argparse import ArgumentParser

import numpy as np
from farms_core.io.yaml import read_yaml
from farms_network.core.network import PyNetwork
from farms_network.core.options import NetworkOptions
from farms_network.gui.gui import NetworkGUI
from imgui_bundle import imgui, imgui_ctx, implot
from tqdm import tqdm


# From farms_amphibious. To be replaced!
def rotate(vector, theta):
    """Rotate vector"""
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation = np.array(((cos_t, -sin_t), (sin_t, cos_t)))
    return np.dot(rotation, vector)


def direction(vector1, vector2):
    """Unit direction"""
    return (vector2-vector1)/np.linalg.norm(vector2-vector1)


def connect_positions(source, destination, dir_shift, perp_shift):
    """Connect positions"""
    connection_direction = direction(source, destination)
    connection_perp = rotate(connection_direction, 0.5*np.pi)
    new_source = (
        source
        + dir_shift*connection_direction
        + perp_shift*connection_perp
    )
    new_destination = (
        destination
        - dir_shift*connection_direction
        + perp_shift*connection_perp
    )
    return new_source, new_destination


def compute_phases(times, data):
    phases = (np.array(data) > 0.1).astype(np.int16)
    # phases = np.logical_not(phases).astype(np.int16)
    phases_xs = []
    phases_ys = []
    for j in range(len(data)):
        phases_start = np.where(np.diff(phases[j, :], prepend=0) == 1.0)[0]
        phases_ends = np.where(np.diff(phases[j, :], append=0) == -1.0)[0]
        phases_xs.append(np.vstack(
            (times[phases_start], times[phases_start], times[phases_ends], times[phases_ends])
        ).T)
        phases_ys.append(np.ones(np.shape(phases_xs[j]))*j)
        # if np.all(len(phases_start) > 3):
        phases_ys[j][:, 1] += 1
        phases_ys[j][:, 2] += 1

    return phases_xs, phases_ys


def add_plot(iteration, data):
    """ """
    # times = data.times.array[iteration%1000:]
    side = "right"
    limb = "fore"
    plot_names = [
        f"{side}_{limb}_RG_E",
        f"{side}_{limb}_RG_F",
        f"left_fore_RG_F",
        f"right_hind_RG_F",
        f"left_hind_RG_F",
        f"{side}_{limb}_PF_FA",
        f"{side}_{limb}_PF_EA",
        f"{side}_{limb}_PF_FB",
        f"{side}_{limb}_PF_EB",
        f"{side}_{limb}_RG_F_DR",
    ]

    plot_labels = [
        "RH_RG_E",
        "RH_RG_F",
        "LF_RG_F",
        "RH_RG_F",
        "LH_RG_F",
        "RH_PF_FA",
        "RH_PF_EA",
        "RH_PF_FB",
        "RH_PF_EB",
        "RH_RG_F_DR",
    ]

    nodes_names = [
        node.name
        for node in data.nodes
    ]

    plot_nodes = [
        nodes_names.index(name)
        for name in plot_names
        if name in nodes_names
    ]
    if not plot_nodes:
        return

    outputs = np.vstack(
        (
            *[
                data.nodes[plot_nodes[j]].output.array
                for j in range(len(plot_nodes))
            ],
            data.nodes[plot_nodes[-1]].external_input.array,
        )
    )
    if iteration < 1000:
        plot_data = np.array(outputs[:, :iteration])
    else:
        plot_data = np.array(outputs[:, iteration-1000:iteration])
    # plot_data = np.vstack((outputs[iteration%1000:], outputs[:iteration%1000]))

    times = np.array((np.linspace(0.0, 1.0, 1000)*-1.0)[::-1])

    phases_xs, phases_ys = compute_phases(times, plot_data[1:5, :])

    # phases = (np.array(plot_data[0, :]) > 0.1).astype(np.int16)
    # phases = np.logical_not(phases).astype(np.int16)
    # phases_start = np.where(np.diff(phases, prepend=0) == 1.0)[0]
    # phases_ends = np.where(np.diff(phases, append=0) == -1.0)[0]
    # phases_xs = np.vstack(
    #     (times[phases_start], times[phases_start], times[phases_ends], times[phases_ends])
    # ).T
    # phases_ys = np.ones(np.shape(phases_xs))
    # if len(phases_start) > 3:
    #     phases_ys[:, 1] += 1.0
    #     phases_ys[:, 2] += 1.0

    colors = {
        "RF": imgui.IM_COL32(28, 107, 180, 255),
        "LF": imgui.IM_COL32(23, 163, 74, 255),
        "RH": imgui.IM_COL32(200, 38, 39, 255),
        "LH":  imgui.IM_COL32(255, 252, 212, 255),  # imgui.IM_COL32(0, 0, 0, 255),
        "right_fore_RG_F": imgui.IM_COL32(28, 107, 180, 255),
        "left_fore_RG_F": imgui.IM_COL32(23, 163, 74, 255),
        "right_hind_RG_F": imgui.IM_COL32(200, 38, 39, 255),
        "left_hind_RG_F": imgui.IM_COL32(255, 252, 212, 255), #  imgui.IM_COL32(0, 0, 0, 255),
    }
    with imgui_ctx.begin("States"):
        if implot.begin_subplots(
                "Network Activity",
                3,
                1,
                imgui.ImVec2(-1, -1),
                row_col_ratios=implot.SubplotsRowColRatios(row_ratios=[0.1, 0.8, 0.1], col_ratios=[1])
        ):
            if implot.begin_plot(""):
                flags = (
                    implot.AxisFlags_.no_label | implot.AxisFlags_.no_tick_labels | implot.AxisFlags_.no_tick_marks
                )
                implot.setup_axis(implot.ImAxis_.y1, "Drive")
                implot.setup_axis(implot.ImAxis_.x1, flags=flags)
                implot.setup_axis_links(implot.ImAxis_.x1, -1.0, 0.0)
                implot.setup_axis_limits(implot.ImAxis_.x1, -1.0, 0.0)
                implot.setup_axis_limits(implot.ImAxis_.y1, 0.0, 1.5)
                implot.setup_axis_limits_constraints(implot.ImAxis_.x1, -1.0, 0.0)
                implot.setup_axis_limits_constraints(implot.ImAxis_.y1, 0.0, 1.5)
                implot.plot_line("RG-F-Dr", times, plot_data[-1, :])
                implot.end_plot()
            if implot.begin_plot(""):
                implot.setup_axis(implot.ImAxis_.y1, "Activity")
                implot.setup_axis(
                    implot.ImAxis_.x1,
                    flags=(
                        implot.AxisFlags_.no_tick_labels |
                        implot.AxisFlags_.no_tick_marks
                    )
                )
                implot.setup_axis_links(implot.ImAxis_.x1, -1.0, 0.0)
                implot.setup_axis_limits(implot.ImAxis_.y1, -1*len(plot_names), 1.0)
                implot.setup_axis_limits_constraints(implot.ImAxis_.x1, -1.0, 0.0)
                implot.setup_axis_limits_constraints(implot.ImAxis_.y1, -8.2, 1.0)
                implot.setup_axis_ticks(
                    axis=implot.ImAxis_.y1,
                    v_min=-8.0,
                    v_max=0.0,
                    n_ticks=int(len(plot_names[:-1])),
                    labels=(plot_labels[:-1])[::-1],
                    keep_default=False
                )
                for j in range(len(plot_nodes[:-1])):
                    if plot_names[j] in colors:
                        implot.push_style_color(implot.Col_.line, colors.get(plot_names[j]))
                        implot.plot_line(plot_names[j], times, plot_data[j, :] - j)
                        implot.pop_style_color()
                    else:
                        implot.plot_line(plot_names[j], times, plot_data[j, :] - j)
                implot.end_plot()
            if len(plot_nodes) > 7:
                if implot.begin_plot(""):
                    implot.setup_axis_limits(implot.ImAxis_.x1, -1.0, 0.0)
                    implot.setup_axis_limits(implot.ImAxis_.y1, 0.0, 4.0)
                    implot.setup_axis_limits_constraints(implot.ImAxis_.x1, -1.0, 0.0)
                    implot.setup_axis_limits_constraints(implot.ImAxis_.y1, 0.0, 4.0)
                    implot.setup_axis(implot.ImAxis_.y1, flags=implot.AxisFlags_.invert)
                    implot.setup_axis_ticks(
                        axis=implot.ImAxis_.y1,
                        v_min=0.5,
                        v_max=3.5,
                        n_ticks=int(4),
                        labels=("RF", "LF", "RH", "LH"),
                        keep_default=False
                    )
                    for j, limb in enumerate(("RF", "LF", "RH", "LH")):
                        # if len(phases_xs[j]) > 3:
                        implot.push_style_color(
                            implot.Col_.fill,
                            colors[limb]
                        )
                        implot.plot_shaded(
                            limb,
                            phases_xs[j].flatten(),
                            phases_ys[j].flatten(),
                            yref=j
                        )
                        implot.pop_style_color()
                    implot.end_plot()
            implot.end_subplots()


def draw_muscle_activity(iteration, data, plot_nodes, plot_names, title):

    outputs = np.vstack(
        [
            data.nodes[plot_nodes[j]].output.array
            for j in range(len(plot_nodes))
        ]
    )
    if iteration < 1000:
        plot_data = np.array(outputs[:, :iteration])
    else:
        plot_data = np.array(outputs[:, iteration-1000:iteration])

    times = np.array((np.linspace(0.0, 1.0, 1000)*-1.0)[::-1])

    with imgui_ctx.begin(title):
        if implot.begin_plot("Muscle Activity", imgui.ImVec2(-1, -1)):
            implot.setup_axis(implot.ImAxis_.x1, "Time")
            implot.setup_axis(implot.ImAxis_.y1, "Activity")
            implot.setup_axis_links(implot.ImAxis_.x1, -1.0, 0.0)
            implot.setup_axis_limits(implot.ImAxis_.x1, -1.0, 0.0)
            implot.setup_axis_limits(implot.ImAxis_.y1, -1*(len(plot_nodes)-1), 1.0)
            implot.setup_axis_limits_constraints(implot.ImAxis_.x1, -1.0, 0.0)
            implot.setup_axis_ticks(
                axis=implot.ImAxis_.y1,
                v_min=-1*(len(plot_nodes)-1),
                v_max=0.0,
                n_ticks=int(len(plot_names)),
                labels=plot_names[::-1],
                keep_default=False
            )
            # implot.setup_axis_limits_constraints(implot.ImAxis_.y1, -5.2, 1.0)
            for j in range(len(plot_nodes)):
                implot.plot_line(plot_names[j], times, plot_data[j, :] - j)
            implot.end_plot()


def plot_hind_motor_activity(iteration, data, side="right"):
    side = "right"
    limb = "hind"

    muscle_names = [
        "bfa",
        "ip",
        "bfpst",
        "rf",
        "va",
        "mg",
        "sol",
        "ta",
        "ab",
        "gm_dorsal",
        "edl",
        "fdl",
    ]

    nodes_names = [
        node.name
        for node in data.nodes
    ]

    plot_nodes = [
        nodes_names.index(f"{side}_{limb}_{name}_Mn")
        for name in muscle_names
        if f"{side}_{limb}_{name}_Mn" in nodes_names
    ]
    draw_muscle_activity(iteration, data, plot_nodes, muscle_names, title="Hindlimb muscles")


def plot_fore_motor_activity(iteration, data, side="right"):
    side = "right"
    limb = "fore"

    muscle_names = [
        "spd",
        "ssp",
        "abd",
        "add",
        "tbl",
        "tbo",
        "bbs",
        "bra",
        "eip",
        "fcu",
    ]

    nodes_names = [
        node.name
        for node in data.nodes
    ]

    plot_nodes = [
        nodes_names.index(f"{side}_{limb}_{name}_Mn")
        for name in muscle_names
        if f"{side}_{limb}_{name}_Mn" in nodes_names
    ]

    draw_muscle_activity(iteration, data, plot_nodes, muscle_names, title="Forelimb muscles")


def __draw_muscle_activity(iteration, data):
    """ Draw muscle activity """
    side = "left"
    limb = "hind"

    muscle_names = [
        "bfa",
        "ip",
        "bfpst",
        "rf",
        "va",
        "mg",
        "sol",
        "ta",
        "ab",
        "gm_dorsal",
        "edl",
        "fdl",
    ]

    nodes_names = [
        node.name
        for node in data.nodes
    ]

    plot_nodes = [
        nodes_names.index(f"{side}_{limb}_{name}_Mn")
        for name in muscle_names
    ]
    if not plot_nodes:
        return
    outputs = np.vstack(
        [
            data.nodes[plot_nodes[j]].output
            for j in range(len(plot_nodes))
        ]
    )
    if iteration < 1000:
        plot_data = np.array(outputs[:, :iteration])
    else:
        plot_data = np.array(outputs[:, iteration-1000:iteration])

    times = np.array((np.linspace(0.0, 1.0, 1000)*-1.0)[::-1])

    with imgui_ctx.begin("Muscle activity"):
        if implot.begin_plot("Muscle Activity", imgui.ImVec2(-1, -1)):
            implot.setup_axis(implot.ImAxis_.x1, "Time")
            implot.setup_axis(implot.ImAxis_.y1, "Activity")
            implot.setup_axis_links(implot.ImAxis_.x1, -1.0, 0.0)
            implot.setup_axis_limits(implot.ImAxis_.x1, -1.0, 0.0)
            implot.setup_axis_limits(implot.ImAxis_.y1, -1*len(plot_nodes), 1.0)
            implot.setup_axis_limits_constraints(implot.ImAxis_.x1, -1.0, 0.0)
            # implot.setup_axis_limits_constraints(implot.ImAxis_.y1, -5.2, 1.0)
            for j in range(len(plot_nodes)):
                implot.plot_line(muscle_names[j], times, plot_data[j, :] - j)
            implot.end_plot()

    # plot_nodes = [
    #     nodes_names.index(f"{side}_{limb}_{name}_Rn")
    #     for name in muscle_names
    # ]
    # if not plot_nodes:
    #     return
    # outputs = np.vstack(
    #     [
    #         data.nodes[plot_nodes[j]].output
    #         for j in range(len(plot_nodes))
    #     ]
    # )
    # if iteration < 1000:
    #     plot_data = np.array(outputs[:, :iteration])
    # else:
    #     plot_data = np.array(outputs[:, iteration-1000:iteration])

    # with imgui_ctx.begin("Renshaw activity"):
    #     if implot.begin_plot("Renshaw Activity", imgui.ImVec2(-1, -1)):
    #         implot.setup_axis(implot.ImAxis_.x1, "Time")
    #         implot.setup_axis(implot.ImAxis_.y1, "Activity")
    #         implot.setup_axis_links(implot.ImAxis_.x1, -1.0, 0.0)
    #         implot.setup_axis_limits(implot.ImAxis_.x1, -1.0, 0.0)
    #         implot.setup_axis_limits(implot.ImAxis_.y1, -1*len(plot_nodes), 1.0)
    #         implot.setup_axis_limits_constraints(implot.ImAxis_.x1, -1.0, 0.0)
    #         # implot.setup_axis_limits_constraints(implot.ImAxis_.y1, -5.2, 1.0)
    #         for j in range(len(plot_nodes)):
    #             implot.plot_line(muscle_names[j], times, plot_data[j, :] - j)
    #         implot.end_plot()

    Ia_In_names = ("EA", "EB", "FA", "FB")
    plot_nodes = [
        nodes_names.index(f"{side}_{limb}_Ia_In_{name}")
        for name in ("EA", "EB", "FA", "FB")
    ]
    plot_nodes = [
        nodes_names.index(name)
        for name in nodes_names
        if "Ib_In_e" in name
    ]
    plot_labels = [
        name
        for name in nodes_names
        if "Ib_In_e" in name
    ]
    if not plot_nodes:
        return
    outputs = np.vstack(
        [
            data.nodes[plot_nodes[j]].output
            for j in range(len(plot_nodes))
        ]
    )
    if iteration < 1000:
        plot_data = np.array(outputs[:, :iteration])
    else:
        plot_data = np.array(outputs[:, iteration-1000:iteration])

    with imgui_ctx.begin("Sensory interneuron activity"):
        if implot.begin_plot("Sensory interneuron Activity", imgui.ImVec2(-1, -1)):
            implot.setup_axis(implot.ImAxis_.x1, "Time")
            implot.setup_axis(implot.ImAxis_.y1, "Activity")
            implot.setup_axis_links(implot.ImAxis_.x1, -1.0, 0.0)
            implot.setup_axis_limits(implot.ImAxis_.x1, -1.0, 0.0)
            implot.setup_axis_limits(implot.ImAxis_.y1, -1*len(plot_nodes), 1.0)
            implot.setup_axis_limits_constraints(implot.ImAxis_.x1, -1.0, 0.0)
            # implot.setup_axis_limits_constraints(implot.ImAxis_.y1, -5.2, 1.0)
            for j in range(len(plot_nodes)):
                implot.plot_line(plot_labels[j], times, plot_data[j, :] - j)
            implot.end_plot()


def draw_vn_activity(iteration, data):
    """ Draw muscle activity """
    side = "left"
    limb = "hind"

    vn_names = [
        f"{side}_{rate}_{axis}_{direction}_In_Vn"
        for rate in ("position", "velocity")
        for direction in ("clock", "cclock")
        for axis in ("pitch", "roll")
        for side in ("left", "right")
    ]

    nodes_names = [
        node.name
        for node in data.nodes
    ]

    plot_nodes = [
        nodes_names.index(name)
        for name in vn_names
    ]
    if not plot_nodes:
        return
    outputs = np.vstack(

        [
            data.nodes[plot_nodes[j]].output.array
            for j in range(len(plot_nodes))
        ]
    )
    if iteration < 1000:
        plot_data = np.array(outputs[:, :iteration])
    else:
        plot_data = np.array(outputs[:, iteration-1000:iteration])

    times = np.array((np.linspace(0.0, 1.0, 1000)*-1.0)[::-1])

    with imgui_ctx.begin("Vestibular"):
        if implot.begin_plot("Vestibular Activity", imgui.ImVec2(-1, -1)):
            implot.setup_axis(implot.ImAxis_.x1, "Time")
            implot.setup_axis(implot.ImAxis_.y1, "Activity")
            implot.setup_axis_links(implot.ImAxis_.x1, -1.0, 0.0)
            implot.setup_axis_limits(implot.ImAxis_.x1, -1.0, 0.0)
            implot.setup_axis_limits(implot.ImAxis_.y1, -1*len(plot_nodes), 1.0)
            implot.setup_axis_limits_constraints(implot.ImAxis_.x1, -1.0, 0.0)
            # implot.setup_axis_limits_constraints(implot.ImAxis_.y1, -5.2, 1.0)
            for j in range(len(plot_nodes)):
                implot.plot_line(vn_names[j], times, plot_data[j, :] - j)
            implot.end_plot()


def draw_network(network_options, data, iteration, edges_x, edges_y):
    """ Draw network """

    nodes = network_options.nodes
    edges = network_options.edges

    imgui.WindowFlags_
    with imgui_ctx.begin("Full-Network"):
        flags = (
            implot.AxisFlags_.no_label |
            implot.AxisFlags_.no_tick_labels |
            implot.AxisFlags_.no_tick_marks
        )
        if implot.begin_plot(
                "vis", imgui.ImVec2((-1, -1)), implot.Flags_.equal
        ):
            implot.setup_axis(implot.ImAxis_.x1, flags=flags)
            implot.setup_axis(implot.ImAxis_.y1, flags=flags)
            implot.plot_line(
                "",
                xs=edges_x,
                ys=edges_y,
                flags=implot.LineFlags_.segments
            )
            radius = 0.1
            circ_x = radius*np.cos(np.linspace(-np.pi, np.pi, 50))
            circ_y = radius*np.sin(np.linspace(-np.pi, np.pi, 50))
            for index, node in enumerate(nodes):
                implot.set_next_marker_style(
                    size=10.0 # *node.visual.radius
                )
                implot.push_style_var(
                    implot.StyleVar_.fill_alpha,
                    0.05+data.nodes[index].output.array[iteration]*3.0
                )
                implot.plot_scatter(
                    "##",
                    xs=np.array((node.visual.position[0],)),
                    ys=np.array((node.visual.position[1],)),
                )
                # implot.plot_line(
                #     "##",
                #     node.visual.position[0]+circ_x,
                #     node.visual.position[1]+circ_y
                # )
                implot.pop_style_var()
                # implot.push_plot_clip_rect()
                # position = implot.plot_to_pixels(implot.Point(node.visual.position[:2]))
                # radius = implot.plot_to_pixels(0.001, 0.001)
                # color = imgui.IM_COL32(255, 0, 0, 255)
                # implot.get_plot_draw_list().add_circle(position, radius[0], color)
                # implot.pop_plot_clip_rect()

                # implot.push_plot_clip_rect()
                # color = imgui.IM_COL32(
                #     100, 185, 0,
                #     int(255*(data.nodes[index].output[iteration]))
                # )
                # implot.get_plot_draw_list().add_circle_filled(position, 7.5, color)
                # implot.pop_plot_clip_rect()
                implot.plot_text(
                    node.visual.label.replace("\\textsubscript", "")[0],
                    node.visual.position[0],
                    node.visual.position[1],
                )

            implot.end_plot()


def draw_slider(
        label: str,
        name: str,
        values: list,
        min_value: float = 0.0,
        max_value: float = 1.0
):
    with imgui_ctx.begin(name):
        clicked, values[0] = imgui.slider_float(
            label="drive",
            v=values[0],
            v_min=min_value,
            v_max=max_value,
        )
        clicked, values[1] = imgui.slider_float(
            label="Ia",
            v=values[1],
            v_min=min_value,
            v_max=max_value,
        )
        clicked, values[2] = imgui.slider_float(
            label="II",
            v=values[2],
            v_min=min_value,
            v_max=max_value,
        )
        clicked, values[3] = imgui.slider_float(
            label="Ib",
            v=values[3],
            v_min=min_value,
            v_max=max_value,
        )
        clicked, values[4] = imgui.slider_float(
            label="Vn",
            v=values[4],
            v_min=-1.0,
            v_max=max_value,
        )
        clicked, values[5] = imgui.slider_float(
            label="Cut",
            v=values[5],
            v_min=min_value,
            v_max=max_value,
        )
    return values


def draw_table(network_options, network_data):
    """ Draw table """
    flags = (
        imgui.TableFlags_.borders | imgui.TableFlags_.row_bg | imgui.TableFlags_.resizable | \
        imgui.TableFlags_.sortable
    )
    with imgui_ctx.begin("Table"):
        edges = network_options.edges
        nodes = network_options.nodes
        n_edges = len(edges)
        if imgui.begin_table("Edges", 3, flags):
            weights = network_data.connectivity.weights
            for col in ("Source", "Target", "Weight"):
                imgui.table_setup_column(col)
            imgui.table_headers_row()
            for row in range(n_edges):
                imgui.table_next_row()
                imgui.table_set_column_index(0)
                imgui.text(edges[row].source)
                imgui.table_set_column_index(1)
                imgui.text(edges[row].target)
                imgui.table_set_column_index(2)
                imgui.push_id(row)
                _, weights[row] = imgui.input_float("##row", weights[row])
                imgui.pop_id()
            imgui.end_table()


def draw_play_pause_button(button_state):
    """ Draw button """

    button_title = "Pause" if button_state else "Play"
    with imgui_ctx.begin("Controls"):
        if imgui.button(button_title):
            button_state = not button_state
            print(button_state)
    return button_state


def main():
    """ Main """

    parser = ArgumentParser()
    parser.add_argument(
        "--config_path", "-c", dest="config_path", type=str, required=True
    )
    clargs = parser.parse_args()
    # run network
    network_options = NetworkOptions.from_options(read_yaml(clargs.config_path))

    network = PyNetwork.from_options(network_options)
    network.setup_integrator(network_options)

    # Integrate
    N_ITERATIONS = network_options.integration.n_iterations
    TIMESTEP = network_options.integration.timestep
    BUFFER_SIZE = network_options.logs.buffer_size

    gui = NetworkGUI()
    gui.create_context()

    inputs_view = network.data.external_inputs.array
    drive_input = 0.0
    imgui.style_colors_dark()
    implot.style_colors_dark()

    edges_xy = np.array(
        [
            network_options.nodes[node_idx].visual.position[:2]
            for edge in network_options.edges
            for node_idx in (
                    network_options.nodes.index(edge.source),
                    network_options.nodes.index(edge.target),
            )
        ]
    )
    # for index in range(len(edges_xy) - 1):
    #     edges_xy[index], edges_xy[index + 1] = connect_positions(
    #         edges_xy[index+1], edges_xy[index], 0.1, 0.0
    #     )
    edges_x = np.array(edges_xy[:, 0])
    edges_y = np.array(edges_xy[:, 1])

    fps = 30.0
    _time_draw = time.time()
    _time_draw_last = _time_draw
    _realtime = 0.1

    imgui.get_io().config_flags |= imgui.ConfigFlags_.docking_enable
    imgui.get_style().anti_aliased_lines = True
    imgui.get_style().anti_aliased_lines_use_tex = True
    imgui.get_style().anti_aliased_fill = True

    drive_input_indices = [
        index
        for index, node in enumerate(network_options.nodes)
        if "DR" in node.name
    ]
    Ia_input_indices = [
        index
        for index, node in enumerate(network_options.nodes)
        if "Ia" == node.name[-2:]
    ]
    II_input_indices = [
        index
        for index, node in enumerate(network_options.nodes)
        if "II" == node.name[-2:]
    ]
    Ib_input_indices = [
        index
        for index, node in enumerate(network_options.nodes)
        if "Ib" == node.name[-2:]
    ]
    Vn_input_indices = [
        index
        for index, node in enumerate(network_options.nodes)
        if "Vn" == node.name[-2:] and node.model == "external_relay"
    ]
    Cut_input_indices = [
        index
        for index, node in enumerate(network_options.nodes)
        if "cut" == node.name[-3:] and node.model == "external_relay"
    ]
    slider_values = np.zeros((6,))
    slider_values[0] = 0.25
    input_array = np.zeros(np.shape(inputs_view))
    input_array[drive_input_indices] = 0.25
    button_state = False
    # input_array[drive_input_indices[0]] *= 1.05

    for iteration in tqdm(range(0, N_ITERATIONS), colour="green", ascii=" >="):
        input_array[drive_input_indices] = slider_values[0]
        input_array[Ia_input_indices] = slider_values[1]
        input_array[II_input_indices] = slider_values[2]
        input_array[Ib_input_indices] = slider_values[3]
        input_array[Vn_input_indices] = slider_values[4]
        input_array[Cut_input_indices] = slider_values[5]

        inputs_view[:] = input_array
        network.step()
        buffer_iteration = iteration%BUFFER_SIZE
        network.data.times.array[buffer_iteration] = (iteration)*TIMESTEP
        _time_draw_last = _time_draw
        _time_draw = time.time()
        fps = _realtime*1/(_time_draw-_time_draw_last)+(1-_realtime)*fps
        # print(imgui.get_io().delta_time, imgui.get_io().framerate)
        implot.push_style_var(implot.StyleVar_.line_weight, 2.0)
        if not (iteration % 1):
            time.sleep(1e-4)
            gui.new_frame()
            slider_values = draw_slider(label="d", name="Drive", values=slider_values)
            add_plot(buffer_iteration, network.data)
            # button_state = draw_play_pause_button(button_state)
            draw_table(network_options, network.data)
            draw_network(network_options, network.data, buffer_iteration, edges_x, edges_y)
            plot_hind_motor_activity(buffer_iteration, network.data)
            plot_fore_motor_activity(buffer_iteration, network.data)
            # draw_vn_activity(buffer_iteration, network.data)
            gui.render_frame()
        implot.pop_style_var()


if __name__ == '__main__':
    main()
