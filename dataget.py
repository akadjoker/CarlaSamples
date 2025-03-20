import carla
import cv2
import numpy as np
import time
import random
import sys
import os
import datetime
import math
import argparse

# Variables for controlling data collection
is_recording = False
collecting_dataset = False
dataset_dir = None
dataset_images_dir = None
dataset_file = None
frame_count = 0
current_image = None

# Display settings
WIDTH = 800
HEIGHT = 600
FPS = 30

def create_dataset_session():
    """Creates a new session to store data (images and steering values)"""
    global dataset_dir, dataset_images_dir, frame_count, dataset_file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_dir = f"dataset/session_{timestamp}"
    dataset_images_dir = f"{dataset_dir}/images"
    os.makedirs(dataset_images_dir, exist_ok=True)
    
    dataset_file = open(f"{dataset_dir}/steering_data.csv", "w")
    dataset_file.write("image_path,steering\n")
    
    frame_count = 0
    print(f"New dataset session created: {dataset_dir}")
    return dataset_dir

def toggle_dataset_collection():
    """Toggle data collection on/off"""
    global collecting_dataset, dataset_file
    if not collecting_dataset:
        os.makedirs("dataset", exist_ok=True)
        create_dataset_session()
        collecting_dataset = True
        print("Starting data collection")
    else:
        if dataset_file:
            dataset_file.close()
            dataset_file = None
        collecting_dataset = False
        print(f"Data collection finished. Total frames: {frame_count}")

def save_frame_to_dataset(frame, steering):
    """Saves a single frame with its steering value to the dataset"""
    global dataset_file, frame_count
    if not dataset_file:
        print("Data collection is off.")
        return False
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_filename = f"frame_{timestamp}.jpg"
    image_path = f"{dataset_images_dir}/{image_filename}"
    cv2.imwrite(image_path, frame)
    
    dataset_file.write(f"images/{image_filename},{steering:.6f}\n")
    dataset_file.flush()
    
    frame_count += 1
    if frame_count % 10 == 0:
        print(f"Frames captured: {frame_count}", end="\r")
    
    return True

def get_road_waypoints(world, vehicle):
    """Gets a series of waypoints that form a complete road segment"""
    map = world.get_map()
    
    # Start from vehicle's current position
    vehicle_location = vehicle.get_location()
    current_waypoint = map.get_waypoint(vehicle_location)
    
    # Get waypoints on the same road
    waypoints = [current_waypoint]
    
    # Collect waypoints forward
    next_waypoint = current_waypoint
    for i in range(100):  # Collect up to 100 waypoints
        next_waypoints = next_waypoint.next(5.0)  # 5 meters between waypoints
        if not next_waypoints:
            break
        next_waypoint = next_waypoints[0]
        waypoints.append(next_waypoint)
    
    print(f"Collected {len(waypoints)} waypoints")
    return waypoints

def process_image(image):
    """Callback function to process camera images"""
    global current_image
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    current_image = array[:, :, :3].copy()

def drive_waypoints(world, vehicle, waypoints, direction_forward=True, speed_factor=0.5):
    """
    Drives the vehicle following the defined waypoints.
    Data capture happens manually when user presses Enter.
    
    Parameters:
    - world: CARLA world object
    - vehicle: Vehicle actor to control
    - waypoints: List of waypoints to follow
    - direction_forward: If False, the route is followed in reverse
    - speed_factor: Controls vehicle speed (0.1-1.0)
    
    Returns:
    - True if route completed, False if user exited
    """
    global current_image
    
    # If direction is reversed, flip the waypoints
    if not direction_forward:
        waypoints = waypoints[::-1]
    
    # Initial settings
    current_waypoint_index = 0
    total_waypoints = len(waypoints)
    vehicle_control = carla.VehicleControl()
    
    # Main loop
    running = True
    paused = False
    
    print("\n=== NAVIGATION STARTED ===")
    print(f"Direction: {'FORWARD' if direction_forward else 'REVERSE'}")
    print("ENTER: Capture frame | P: Pause | +/-: Speed | ESC: Exit")
    
    while running and current_waypoint_index < total_waypoints:
        # Update the world in synchronous mode
        world.tick()
        
        # Get the next waypoint
        target_waypoint = waypoints[current_waypoint_index]
        
        if not paused:
            # Calculate vehicle and waypoint positions
            vehicle_transform = vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            target_location = target_waypoint.transform.location
            
            # Calculate direction and distance
            direction = target_location - vehicle_location
            distance = math.sqrt(direction.x**2 + direction.y**2)
            
            # If close enough, advance to the next waypoint
            if distance < 2.0:  # 2 meters tolerance
                current_waypoint_index += 1
                if current_waypoint_index % 10 == 0:
                    print(f"Waypoint {current_waypoint_index}/{total_waypoints}")
                if current_waypoint_index >= total_waypoints:
                    print("Route completed!")
                    break
                continue
            
            # Calculate angle between vehicle and waypoint
            vehicle_forward = vehicle_transform.get_forward_vector()
            dot = vehicle_forward.x * direction.x + vehicle_forward.y * direction.y
            cross = vehicle_forward.x * direction.y - vehicle_forward.y * direction.x
            angle = math.atan2(cross, dot)
            
            # Convert angle to steering value [-1, 1]
            steering = max(-1.0, min(1.0, angle * 2.0))
            
            # Apply vehicle control
            vehicle_control.throttle = speed_factor
            vehicle_control.steer = steering
            vehicle_control.brake = 0.0
            vehicle.apply_control(vehicle_control)
        
        # Show images
        if current_image is not None:
            img_display = current_image.copy()
            
            # Add information to the image
            velocity = vehicle.get_velocity()
            speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h
            current_steering = vehicle_control.steer
            
            # Basic information
            cv2.putText(img_display, f"Speed: {speed:.1f} km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_display, f"Steering: {current_steering:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_display, f"Waypoint: {current_waypoint_index}/{total_waypoints}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_display, f"Direction: {'FORWARD' if direction_forward else 'REVERSE'}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Dataset information
            if collecting_dataset:
                cv2.putText(img_display, f"RECORDING: {frame_count} frames", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Blinking recording indicator
                if int(time.time() * 2) % 2 == 0:
                    cv2.circle(img_display, (25, 150), 10, (0, 0, 255), -1)
            
            # Navigation status
            if paused:
                cv2.putText(img_display, "NAVIGATION PAUSED (press 'R' to continue)", 
                           (10, HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Speed factor
            cv2.putText(img_display, f"Speed factor: {speed_factor:.2f}", 
                       (WIDTH - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Commands
            cv2.putText(img_display, "ENTER: Capture | T: New session | +/-: Speed | ESC: Exit", 
                       (10, HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('CARLA - Navigation', img_display)
            
            # Process keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                running = False
                print("Navigation interrupted by user.")
                return False
            elif key == 13:  # ENTER
                # Capture frame for the dataset
                if not collecting_dataset:
                    toggle_dataset_collection()
                
                if save_frame_to_dataset(current_image.copy(), current_steering):
                    print(f"Frame captured! Total: {frame_count}")
            elif key == ord('t'):
                # Create new dataset session
                if collecting_dataset:
                    toggle_dataset_collection()
                toggle_dataset_collection()
            elif key == ord('p') or key == ord('r'):
                # Pause/resume navigation
                paused = not paused if key == ord('p') else False
                print(f"Navigation {'paused' if paused else 'resumed'}.")
                if paused:
                    # Stop the vehicle
                    vehicle_control.throttle = 0.0
                    vehicle_control.brake = 1.0
                    vehicle.apply_control(vehicle_control)
            elif key == ord('+') or key == ord('='):
                # Increase speed
                speed_factor = min(1.0, speed_factor + 0.05)
                print(f"Speed increased to: {speed_factor:.2f}")
            elif key == ord('-') or key == ord('_'):
                # Decrease speed
                speed_factor = max(0.1, speed_factor - 0.05)
                print(f"Speed decreased to: {speed_factor:.2f}")
    
    # Stop the vehicle at the end of the route
    vehicle_control.throttle = 0.0
    vehicle_control.brake = 1.0
    vehicle.apply_control(vehicle_control)
    time.sleep(1)  # Let the vehicle come to a stop
    
    print("Vehicle stopped.")
    return True  # Route completed normally

def main():
    parser = argparse.ArgumentParser(description='CARLA Continuous Data Collection with Auto-Reverse')
    parser.add_argument('--map', type=str, default='Town04', help='Map to use (Town04 recommended)')
    parser.add_argument('--speed', type=float, default=0.5, help='Initial speed factor (0.1-1.0)')
    parser.add_argument('--loops', type=int, default=0, help='Number of loops to drive (0 = infinite)')
    args = parser.parse_args()
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    # Load the specified map
    try:
        world = client.load_world(args.map)
        print(f"Map loaded: {args.map}")
    except Exception as e:
        print(f"Error loading map: {e}")
        world = client.get_world()
        print(f"Using current map: {world.get_map().name}")
    
    # Reduce rendering complexity to improve performance
    world.unload_map_layer(carla.MapLayer.All)
    world.load_map_layer(carla.MapLayer.Ground)
    world.unload_map_layer(carla.MapLayer.Decals)
    world.unload_map_layer(carla.MapLayer.Props)
    world.unload_map_layer(carla.MapLayer.StreetLights)
    world.unload_map_layer(carla.MapLayer.Foliage)
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)
    world.unload_map_layer(carla.MapLayer.Particles)
    world.unload_map_layer(carla.MapLayer.Walls)
    
    # Enable synchronous mode
    settings = world.get_settings()
    original_settings = settings
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    print("Synchronous mode enabled")
    
    actors_list = []
    
    try:
        # Select blueprint for the vehicle
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        
        # Get spawn points
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            print("No spawn points available!")
            return
        
        # Choose a spawn point that's likely to be on a good road
        spawn_point = random.choice(spawn_points)
        print(f"Selected spawn point: {spawn_point}")
        
        # Create the vehicle
        print("Creating vehicle...")
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actors_list.append(vehicle)
        print(f"Vehicle created: {vehicle.type_id}")
        
        # Configure camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(WIDTH))
        camera_bp.set_attribute('image_size_y', str(HEIGHT))
        camera_bp.set_attribute('fov', '90')
        
        camera_transform = carla.Transform(
            carla.Location(x=2.0, z=1.4),
            carla.Rotation(pitch=-15)
        )
        
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actors_list.append(camera)
        print("Camera created")
        
        # Register callback function for the camera
        camera.listen(process_image)
        
        # Create window for display
        cv2.namedWindow('CARLA - Navigation', cv2.WINDOW_NORMAL)
        
        # Wait a bit for the world to initialize
        for i in range(3, 0, -1):
            print(f"Starting in {i}...")
            time.sleep(1)
            world.tick()
        
        # Main loop for continuous driving with direction switching
        direction_forward = True
        current_speed = args.speed
        loop_count = 0
        
        while True:
            # Find road waypoints from current vehicle position
            print("Finding road waypoints...")
            waypoints = get_road_waypoints(world, vehicle)
            
            if len(waypoints) < 10:
                print("Could not find enough waypoints. Repositioning vehicle...")
                # Try to reposition to a different spawn point
                spawn_point = random.choice(spawn_points)
                vehicle.set_transform(spawn_point)
                time.sleep(1)
                continue
            
            # Drive the current route
            print(f"\n=== {'FORWARD' if direction_forward else 'REVERSE'} DIRECTION (Loop {loop_count+1}) ===")
            
            # This function returns False if the user pressed ESC
            if not drive_waypoints(world, vehicle, waypoints, direction_forward, current_speed):
                break
            
            # Toggle direction for next iteration
            direction_forward = not direction_forward
            loop_count += 0.5  # Count a complete loop as forward + reverse
            
            print(f"\nRoute completed! Automatically switching direction to {('FORWARD' if direction_forward else 'REVERSE')}")
            
            # Check if we've reached the loop limit
            if args.loops > 0 and loop_count >= args.loops:
                print(f"\nCompleted {args.loops} loops as requested.")
                break
    
    finally:
        # Restore original world settings
        world.apply_settings(original_settings)
        
        # Clean up actors
        print("Cleaning up actors...")
        for actor in actors_list:
            if actor is not None and actor.is_alive:
                actor.destroy()
        
        # Close dataset file if open
        if 'dataset_file' in globals() and dataset_file:
            dataset_file.close()
            print(f"Dataset saved with {frame_count} frames")
        
        cv2.destroyAllWindows()
        print("Simulation terminated")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Execution interrupted by user")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error: {e}")
        cv2.destroyAllWindows()