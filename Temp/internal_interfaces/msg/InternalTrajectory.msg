# Trajectory description
# The topic ExtGCU_Trajectory is used to receive the user desired waypoints set from guidance module 
# (Guidance/Obstacle Detection/Collision Avoidance).
# The topic ExtGCU_Trajectory is exclusively used in the Aware4 Project companion device.

uint64 timestamp		            # [us] time since system start

uint8 last_wpt_call                 # [-] last waypoint call signal (binary)
uint8 wpt_update_reply              # [-] waypoint update reply (considering UDP environment)
uint8 external_wpt_manager_ready    # [-] readiness of external waypoint manager

ExtGCUWaypoint[2] waypoints         # [-] three waypoints set (past/current/next)
