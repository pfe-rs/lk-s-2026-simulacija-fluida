import argparse
import csv
from pathlib import Path
import time

import numpy as np
import cupy as cp
import pygame

from kontinuitet import FluidSimulation, PRESETS


def parse_args():
    parser = argparse.ArgumentParser(description="Fast Pygame viewer for the fluid simulation.")
    parser.add_argument("--n", type=int, default=256, help="Grid size.")
    parser.add_argument("--dt", type=float, default=0.01, help="Simulation time step.")
    parser.add_argument("--h", type=float, default=0.1, help="Cell size.")
    parser.add_argument("--viscosity", type=float, default=0.00, help="Fluid viscosity.")
    parser.add_argument(
        "--preset",
        default="four_vortices",
        choices=sorted(set(PRESETS.values())),
        help="Initial velocity field.",
    )
    parser.add_argument(
        "--geometry",
        default="venturi",
        choices=["venturi", "flat"],
        help="Duct geometry.",
    )
    parser.add_argument("--substeps", type=int, default=1, help="Simulation steps per frame.")
    parser.add_argument("--size", type=int, default=768, help="Square simulation viewport size.")
    parser.add_argument(
        "--quiver-stride",
        type=int,
        default=7,
        help="Draw every Nth velocity arrow.",
    )
    parser.add_argument(
        "--quiver", 
        action="store_true", 
        help="Prikaži strelice vektorskog polja brzine (podrazumevano je skriveno)."
    )
    parser.add_argument("--max-fps", type=int, default=60, help="0 means uncapped.")
    parser.add_argument("--frames", type=int, default=0, help="Quit after N frames. 0 means run forever.")
    parser.add_argument(
        "--stream-strength",
        type=float,
        default=6.0,
        help="Per-frame vortex strength while a mouse button is held.",
    )
    parser.add_argument(
        "--stream-radius",
        type=float,
        default=0.45,
        help="Radius of the held mouse stream in simulation units.",
    )
    parser.add_argument(
        "--pressure-scale",
        type=float,
        default=5.0,
        help="Pressure magnitude mapped to full color intensity.",
    )
    parser.add_argument(
        "--view-mode",
        default="pressure",
        choices=["pressure", "curl", "speed"],
        help="biras polje",
    )
    parser.add_argument(
        "--curl-scale",
        type=float,
        default=20.0,  
        help="Rotor",
)   
    
    parser.add_argument(
        "--speed-scale",
        type=float,
        default=20.0,  
        help="Brzina brm brm",
)

    parser.add_argument(
        "--density",
        type=float,
        default=1.0,
        help="Gustina",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="classic",
        choices=["klasikica", "vatra", "emerald", "sajber"],
        help="Izaberi kolor temu za vizuelizaciju fluida"
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=95,
        help="Faktor prigušenja brzine fluida po frejmu (npr. 0.995 za lagano nestajanje).",
    )

    parser.add_argument(
        "--export-csv",
        default="",
        help="Optional CSV path for exported speed and pressure samples.",
    )
    parser.add_argument(
        "--export-profile-csv",
        default="",
        help="Optional CSV path for per-frame x profiles of velocity, speed, and pressure.",
    )
    parser.add_argument(
        "--export-every",
        type=int,
        default=1,
        help="Export one CSV row every N simulation frames.",
    )

    return parser.parse_args()

def open_export_csv(path):
    if not path:
        return None, None

    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = csv_path.open("w", newline="")
    fieldnames = [
        "frame",
        "geometry",
        "preset",
        "time",
        "p_wide",
        "p_narrow",
        "p_delta",
        "speed_wide",
        "speed_narrow",
        "speed_ratio",
        "v1",
        "v2",
        "v3",
        "v3_v1",
        "divergence",
        "vorticity",
        "curl",
        "kinetic_energy",
        "cfl",
        "l2_div",
        "avg_l2_div",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    return csv_file, writer

def open_profile_csv(path):
    if not path:
        return None, None

    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = csv_path.open("w", newline="")
    fieldnames = [
        "frame",
        "geometry",
        "preset",
        "time",
        "x",
        "velocity_x",
        "speed",
        "pressure",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    return csv_file, writer

def draw_help_menu(surface, font, title_font, viewport_size):

    menu_w, menu_h = 550, 550
    menu_x = (viewport_size - menu_w) // 2
    menu_y = (viewport_size - menu_h) // 2
    
    overlay = pygame.Surface((menu_w, menu_h), pygame.SRCALPHA)
    overlay.fill((10, 15, 25, 235))  
    
    pygame.draw.rect(overlay, (0, 180, 255), (0, 0, menu_w, menu_h), 2, border_radius=8)
    
    
    title = title_font.render("KONTROLE SIMULACIJE", True, (0, 180, 255))
    overlay.blit(title, (menu_w // 2 - title.get_width() // 2, 20))
    

    pygame.draw.line(overlay, (40, 50, 70), (30, 55), (menu_w - 30, 55), 1)
    
    kontrole = [
        ("SPACE", "Pauza"),
        ("R", "Reset"),
        ("M", "Metrike"),
        ("ESC", "Izlaz iz aplikacije"),
        ("TASTERI 1 - 6", "Presetovi"),
        ("K, P, S", "Curl, Pressure, Speed mode"),
        ("t", "teme šaltanje"),

    ]
    
    start_y = 75
    for taster, opis in kontrole:
        if taster == "" and opis == "":
            start_y += 15
            continue
            
        
        txt_taster = font.render(taster, True, (240, 245, 255))
        overlay.blit(txt_taster, (30, start_y))
        
        
        txt_opis = font.render(f" -  {opis}", True, (170, 180, 195))
        overlay.blit(txt_opis, (220, start_y))
        
        start_y += 26
        
    
    pygame.draw.line(overlay, (40, 50, 70), (30, menu_h - 45), (menu_w - 30, menu_h - 45), 1)
    footer = font.render("aki i laki", True, (100, 115, 135))
    overlay.blit(footer, (menu_w // 2 - footer.get_width() // 2, menu_h - 32))
    
    surface.blit(overlay, (menu_x, menu_y))

def field_to_rgb(field, scale, theme_name="klasikica", is_vector_magnitude=False, simulation=None):
    rgb = cp.empty((*field.shape, 3), dtype=cp.uint8)
    
    if theme_name == "vatra":
        normalized = cp.clip(field / scale if is_vector_magnitude else cp.abs(field) / scale, 0.0, 1.0)
        rgb[..., 0] = (normalized * 255).astype(cp.uint8)
        rgb[..., 1] = ((normalized ** 2) * 255).astype(cp.uint8)
        rgb[..., 2] = ((normalized ** 4) * 255).astype(cp.uint8)
        
    elif theme_name == "emerald":
        normalized = cp.clip(field / scale if is_vector_magnitude else cp.abs(field) / scale, 0.0, 1.0)
        rgb[..., 0] = ((normalized ** 3) * 150).astype(cp.uint8)
        rgb[..., 1] = (50 + normalized * 205).astype(cp.uint8)
        rgb[..., 2] = ((normalized ** 2) * 100).astype(cp.uint8)
        
    elif theme_name == "sajber":
        normalized = cp.clip(field / scale if is_vector_magnitude else cp.abs(field) / scale, 0.0, 1.0)
        rgb[..., 0] = (normalized * 255).astype(cp.uint8)
        rgb[..., 1] = ((1.0 - normalized) * 30 + (normalized ** 2) * 20).astype(cp.uint8)
        rgb[..., 2] = (150 + (1.0 - normalized) * 105).astype(cp.uint8)
        
    else:  
        normalized = cp.clip(field / scale, -1.0, 1.0)
        positive = cp.clip(normalized, 0.0, 1.0)
        negative = cp.clip(-normalized, 0.0, 1.0)
        
        rgb[..., 0] = (35 + 220 * positive).astype(cp.uint8)
        rgb[..., 1] = (45 + 150 * (1.0 - cp.abs(normalized))).astype(cp.uint8)
        rgb[..., 2] = (55 + 200 * negative).astype(cp.uint8)

    
    if simulation is not None and hasattr(simulation, "cell_type"):
        
        wall_mask = (simulation.cell_type == 0)
        
        rgb[wall_mask] = 0  

    return rgb

def draw_velocity_arrows(surface, simulation, viewport_size, stride):
    u_center, v_center = simulation.centered_velocity()
    
    u_cpu = u_center.get()
    v_cpu = v_center.get()


    cell_size = viewport_size / simulation.n
    scale = cell_size * 0.13



    for i in range(1, simulation.n - 1, stride):
        y = (i + 0.5) * cell_size
        for j in range(1, simulation.n - 1, stride):
            x = (j + 0.5) * cell_size
            dx = float(u_cpu[i, j]) * scale
            dy = float(v_cpu[i, j]) * scale
            end = (x + dx, y + dy)
            pygame.draw.line(surface, (240, 245, 255), (x, y), end, 1)
            pygame.draw.circle(surface, (240, 245, 255), (int(end[0]), int(end[1])), 2)


def draw_text_lines(surface, font, lines, x, y, color=(235, 240, 245)):
    line_height = font.get_height() + 2
    width = max(font.size(line)[0] for line in lines) + 12
    height = len(lines) * line_height + 10
    panel = pygame.Surface((width, height), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 150))
    surface.blit(panel, (x, y))

    for index, line in enumerate(lines):
        rendered = font.render(line, True, color)
        surface.blit(rendered, (x + 6, y + 5 + index * line_height))

def main():
    args = parse_args()
    simulation = FluidSimulation(
        n=args.n,
        dt=args.dt,
        h=args.h,
        viscosity=args.viscosity,
        preset=args.preset,
        geometry=args.geometry,
    )
    export_file, export_writer = open_export_csv(args.export_csv)
    profile_file, profile_writer = open_profile_csv(args.export_profile_csv)

    pygame.init()
    pygame.display.set_caption("Real-time Fluid Simulation - Pygame")
    screen = pygame.display.set_mode((args.size, args.size))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 15)
    small_font = pygame.font.SysFont("consolas", 13)
    title_font = pygame.font.SysFont("consolas", 18, bold=True) 

    paused = False
    running = True
    show_metrics = False
    view_mode = args.view_mode
    fps = 0.0
    timings = {
        "events_ms": 0.0,
        "stream_ms": 0.0,
        "sim_total_ms": 0.0,
        "rgb_ms": 0.0,
        "scale_ms": 0.0,
        "quiver_ms": 0.0,
        "text_ms": 0.0,
        "flip_ms": 0.0,
        "frame_ms": 0.0,
    }
    last_frame_time = time.perf_counter()
    last_step_timings = {
        "rhs_ms": 0.0,
        "pressure_ms": 0.0,
        "projection_ms": 0.0,
        "advection_ms": 0.0,
        "diffusion_ms": 0.0,
        "walls_ms": 0.0,
        "total_ms": 0.0,
    }
    divergencija_metrika = 0.0
    vorticitet_metrika = 0.0
    curl = 0.0
    kinetic_energy = 0.0
    cfl = 0.0
    tacnost_L2 = 0.0
    avg_tacnostL2 = 0.0
    p_wide = 0.0
    p_narrow = 0.0
    p_delta = 0.0
    speed_wide = 0.0
    speed_narrow = 0.0
    speed_ratio = 0.0

    mouse_pressed_pos = None
    mouse_timer = 0

    sve_teme = ["klasikica", "vatra", "emerald", "sajber"]
    tema_idx = 0
    while running:


        frame_start = time.perf_counter()
 
        event_start = time.perf_counter()
        for event in pygame.event.get():


            

            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    simulation.reset()
                elif event.key == pygame.K_m:
                    show_metrics = not show_metrics
                elif event.key == pygame.K_k:
                    view_mode = "curl"    
                elif event.key == pygame.K_s:
                    view_mode = "speed"
                elif event.key == pygame.K_p:
                    view_mode = "pressure"
                elif event.key == pygame.K_t:         
                    if tema_idx != (len(sve_teme) - 1):
                        args.theme = sve_teme[tema_idx + 1]
                        tema_idx += 1
                    else:
                        tema_idx = 0
                        args.theme = sve_teme[tema_idx]
            
                else:
                    key_name = pygame.key.name(event.key)
                    if key_name in PRESETS:
                        simulation.reset(PRESETS[key_name])

        mouse_buttons = pygame.mouse.get_pressed(num_buttons=3)
        current_mouse_pos = pygame.mouse.get_pos()
        screen_size = screen.get_width()

        
        cx = int(current_mouse_pos[0] / (screen_size / simulation.n))
        cy = int(current_mouse_pos[1] / (screen_size / simulation.n))
        cx = max(0, min(simulation.n - 1, cx))
        cy = max(0, min(simulation.n - 1, cy))

        
        if mouse_buttons[0]:
            if mouse_pressed_pos is None:
                
                mouse_pressed_pos = (cx, cy)
                mouse_timer = 0

            else:
                
                mouse_timer += 1
    
        elif mouse_pressed_pos is not None and not mouse_buttons[0]:
            duration = max(1, mouse_timer)

            simulation.apply_mouse_impulse(mouse_pressed_pos, (cx, cy), duration)
            
            
            mouse_pressed_pos = None

        
        if mouse_buttons[2]:
            if 0 <= current_mouse_pos[0] < args.size and 0 <= current_mouse_pos[1] < args.size:
                stream_strength = args.stream_strength * -1.0
                simulation.add_vortex_at_cell(
                    current_mouse_pos[0] / args.size * simulation.n,
                    current_mouse_pos[1] / args.size * simulation.n,
                    radius=args.stream_radius,
                    strength=stream_strength,
                )
        
        
        timings["events_ms"] = (time.perf_counter() - event_start) * 1000.0

        
        stream_start = time.perf_counter()
        stream_strength = 0.0
        

        mouse_buttons = pygame.mouse.get_pressed(num_buttons=3)
        

        if mouse_buttons[2]:
            x, y = pygame.mouse.get_pos()
            if 0 <= x < args.size and 0 <= y < args.size:
                stream_strength = args.stream_strength * -1.0
                simulation.add_vortex_at_cell(
                    x / args.size * simulation.n,
                    y / args.size * simulation.n,
                    radius=args.stream_radius,
                    strength=stream_strength,
                )
        timings["stream_ms"] = (time.perf_counter() - stream_start) * 1000.0

        if not paused:

            last_step_timings = simulation.step(args.substeps, profile=True)
            timings["sim_total_ms"] = last_step_timings["total_ms"]

            simulation.velocity_x *= (args.damping / 100)
            simulation.velocity_y *= (args.damping / 100)

            (
                divergencija_metrika,
                vorticitet_metrika,
                curl,
                kinetic_energy,
                cfl,
                tacnost_L2,
                avg_tacnostL2,
                p_wide,
                p_narrow,
                p_delta,
                speed_wide,
                speed_narrow,
                speed_ratio,
            ) = simulation.metrics()

        rgb_start = time.perf_counter()
        if view_mode == "curl":
            rgb = field_to_rgb(simulation.curl_field(), args.curl_scale, args.theme, False, simulation)
        elif view_mode == "speed":
            rgb = field_to_rgb(simulation.speed(), args.speed_scale, args.theme, True, simulation)
        else:
            rgb = field_to_rgb(simulation.pressure, args.pressure_scale, args.theme, False, simulation)
        timings["rgb_ms"] = (time.perf_counter() - rgb_start) * 1000.0
        timings["rgb_ms"] = (time.perf_counter() - rgb_start) * 1000.0
        timings["rgb_ms"] = (time.perf_counter() - rgb_start) * 1000.0

        scale_start = time.perf_counter()
        cpu_rgb = cp.swapaxes(rgb, 0, 1).get() # Spuštamo sa GPU na CPU
        field_surface = pygame.surfarray.make_surface(cpu_rgb)
        field_surface = pygame.transform.scale(field_surface, (args.size, args.size))
        screen.blit(field_surface, (0, 0))
        timings["scale_ms"] = (time.perf_counter() - scale_start) * 1000.0

        quiver_start = time.perf_counter()

        if args.quiver:
            draw_velocity_arrows(screen, simulation, args.size, max(1, args.quiver_stride))
            
        if stream_strength != 0.0:
            color = (255, 230, 120) if stream_strength > 0.0 else (125, 210, 255)
            pygame.draw.circle(screen, color, pygame.mouse.get_pos(), 12, 2)
            pygame.draw.circle(screen, color, pygame.mouse.get_pos(), 3)
        timings["quiver_ms"] = (time.perf_counter() - quiver_start) * 1000.0

        now = time.perf_counter()
        elapsed = now - last_frame_time
        if elapsed > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / elapsed)
        last_frame_time = now

        text_start = time.perf_counter()
        v1 = simulation.velocity_x[simulation.n//2, simulation.n//6]
        v2 = simulation.velocity_x[simulation.n//2, simulation.n//2]
        v3 = simulation.velocity_x[simulation.n//2, 5 * simulation.n//6]
        v1_value = float(cp.asnumpy(v1))
        v2_value = float(cp.asnumpy(v2))
        v3_value = float(cp.asnumpy(v3))
        v3_v1 = v3_value / v1_value if abs(v1_value) > 1e-8 else 0.0
        if (
            not paused
            and (export_writer is not None or profile_writer is not None)
            and simulation.frame % max(1, args.export_every) == 0
        ):
            if export_writer is not None:
                export_writer.writerow(
                    {
                        "frame": simulation.frame,
                        "geometry": simulation.geometry,
                        "preset": simulation.preset,
                        "time": simulation.frame * simulation.dt,
                        "p_wide": p_wide,
                        "p_narrow": p_narrow,
                        "p_delta": p_delta,
                        "speed_wide": speed_wide,
                        "speed_narrow": speed_narrow,
                        "speed_ratio": speed_ratio,
                        "v1": v1_value,
                        "v2": v2_value,
                        "v3": v3_value,
                        "v3_v1": v3_v1,
                        "divergence": divergencija_metrika,
                        "vorticity": vorticitet_metrika,
                        "curl": curl,
                        "kinetic_energy": kinetic_energy,
                        "cfl": cfl,
                        "l2_div": tacnost_L2,
                        "avg_l2_div": avg_tacnostL2,
                    }
                )
            if profile_writer is not None:
                profile = simulation.profile_samples()
                for x, velocity_x, speed, pressure in zip(
                    profile["x"],
                    profile["velocity_x"],
                    profile["speed"],
                    profile["pressure"],
                ):
                    profile_writer.writerow(
                        {
                            "frame": simulation.frame,
                            "geometry": simulation.geometry,
                            "preset": simulation.preset,
                            "time": simulation.frame * simulation.dt,
                            "x": int(x),
                            "velocity_x": float(velocity_x),
                            "speed": float(speed),
                            "pressure": float(pressure),
                        }
                    )
        status = "paused" if paused else "running"
        lines = [
            f"{simulation.geometry} | {simulation.preset} | frame {simulation.frame} | {status} | fps {fps:5.1f}",
            "simulation timing (ms)",
            f"rhs        {last_step_timings['rhs_ms']:6.2f}",
            f"pressure   {last_step_timings['pressure_ms']:6.2f}",
            f"project    {last_step_timings['projection_ms']:6.2f}",
            f"advect     {last_step_timings['advection_ms']:6.2f}",
            f"diffuse    {last_step_timings['diffusion_ms']:6.2f}",
            f"walls      {last_step_timings['walls_ms']:6.2f}",
            f"sim total  {timings['sim_total_ms']:6.2f}",

            f"BENCHMARK METRIKE",
            f"div metrika {divergencija_metrika:.2f}",
            f"vorticitet metrika {vorticitet_metrika:.2f}",
            f"curl metrika {curl:.2f}",
            f"kinetic energy {kinetic_energy:.2f}",
            f"cfl {cfl:.2f}\n",

            f"L2-div {tacnost_L2}",
            f"L2-div-abf {avg_tacnostL2}",
            "render timing (ms)",
            f"events     {timings['events_ms']:6.2f}",
            f"stream     {timings['stream_ms']:6.2f}",
            f"rgb        {timings['rgb_ms']:6.2f}",
            f"scale      {timings['scale_ms']:6.2f}",
            f"quiver     {timings['quiver_ms']:6.2f}",
            f"text       {timings['text_ms']:6.2f}",
            f"flip       {timings['flip_ms']:6.2f}",
            f"v1         {v1_value:.2f}",
            f"v2         {v2_value:.2f}",
            f"v3         {v3_value:.2f}",
            f"speed wide {speed_wide:.2f}",
            f"speed nar  {speed_narrow:.2f}",
            f"speed rat  {speed_ratio:.2f}",
            f"p wide     {p_wide:.2f}",
            f"p narrow   {p_narrow:.2f}",
            f"delta p    {p_delta:.2f}",
            f"frame      {timings['frame_ms']:6.2f}"      
        ]
        if show_metrics:
            draw_text_lines(screen, font, lines, 10, 10)
            help_lines = ["space pause | r reset | 1-6 presets | hold L/R stream | esc quit"]
            draw_text_lines(screen, small_font, help_lines, 10, args.size - 32)
            timings["text_ms"] = (time.perf_counter() - text_start) * 1000.0
        else :
            lines_small = [ f"{simulation.geometry} | {simulation.preset} | {status} | fps {fps:5.1f}", 
                            f"L2-div {tacnost_L2:.2f}", f"L2-div-abf {avg_tacnostL2:.2f}",
                            f"v3/v1 {v3_v1}",
                            f"dp {p_delta:.2f}",
                            f"Kontrole - TAB"]      
            draw_text_lines(screen, font, lines_small, 5, 5)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_TAB]:
            draw_help_menu(screen, font, title_font, args.size)

        flip_start = time.perf_counter()
        pygame.display.flip()
        timings["flip_ms"] = (time.perf_counter() - flip_start) * 1000.0
        timings["frame_ms"] = (time.perf_counter() - frame_start) * 1000.0

        if args.max_fps > 0:
            clock.tick(args.max_fps)

        if args.frames > 0 and simulation.frame >= args.frames:
            running = False

    if export_file is not None:
        export_file.close()
    if profile_file is not None:
        profile_file.close()
    pygame.quit()

if __name__ == "__main__":
    main()
