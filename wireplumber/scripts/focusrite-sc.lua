-- This file is a script that automatically links SuperCollider
-- to either a Focusrite Scarlett 2i2 or 18i20.
-- It also automatically disconnects SuperCollider from the internal
-- sound card, but only if the Focusrite is present.

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-- Object managers
-- These are used to detect the SuperCollider and interface ports.
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

-- Focusrite
local interface_om = ObjectManager {
  Interest {
    type = "port",
    Constraint { "port.alias", "matches", "Scarlett*" },
  },
}

-- SuperCollider
local supercollider_om = ObjectManager {
  Interest {
    type = "port",
    Constraint { "port.alias", "matches", "SuperCollider*" },
  },
}

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-- Specify connections to make here!
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

local line_in_connections = {
  -- for the 2i2
  {
    output = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*capture_FL" },
    },
    input = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*in_1" },
    },
  },
  {
    output = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*capture_FR" },
    },
    input = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*in_2" },
    },
  },
  -- for the 18i20
  {
    output = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*capture_AUX0" },
    },
    input = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*in_1" },
    },
  },
  {
    output = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*capture_AUX1" },
    },
    input = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*in_2" },
    },
  },
  {
    output = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*capture_AUX2" },
    },
    input = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*in_3" },
    },
  },
  {
    output = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*capture_AUX3" },
    },
    input = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*in_4" },
    },
  },
  {
    output = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*capture_AUX4" },
    },
    input = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*in_5" },
    },
  },
  {
    output = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*capture_AUX5" },
    },
    input = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*in_6" },
    },
  },
  {
    output = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*capture_AUX6" },
    },
    input = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*in_7" },
    },
  },
  {
    output = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*capture_AUX7" },
    },
    input = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*in_8" },
    },
  },
}

local line_out_connections = {
  -- for the 2i2
  {
    output = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*out_1" },
    },
    input = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*playback_FL" },
    },
  },
  {
    output = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*out_2" },
    },
    input = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*playback_FR" },
    },
  },
  -- for the 18i20
  {
    output = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*out_1" },
    },
    input = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*playback_AUX0" },
    },
  },
  {
    output = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*out_2" },
    },
    input = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*playback_AUX1" },
    },
  },
  {
    output = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*out_3" },
    },
    input = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*playback_AUX2" },
    },
  },
  {
    output = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*out_4" },
    },
    input = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*playback_AUX3" },
    },
  },
  {
    output = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*out_5" },
    },
    input = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*playback_AUX4" },
    },
  },
  {
    output = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*out_6" },
    },
    input = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*playback_AUX5" },
    },
  },
  {
    output = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*out_7" },
    },
    input = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*playback_AUX6" },
    },
  },
  {
    output = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*out_8" },
    },
    input = Interest {
      type = "port",
      Constraint { "port.alias", "matches", "*playback_AUX7" },
    },
  },
}

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-- LINKING LOGIC
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

-- Takes two "port" nodes and links them
local function link_ports(node_left, node_right)
  local link = Link("link-factory", {
    ["link.output.node"] = node_left.properties["node.id"],
    ["link.output.port"] = node_left.properties["port.id"],
    ["link.input.node"] = node_right.properties["node.id"],
    ["link.input.port"] = node_right.properties["port.id"],
    ["object.linger"] = true,
  })
  link:activate(1)
end

-- Makes all specified connections.
-- This function is run each time one of the ObjectManagers
-- detects a new object.
local function make_all_connections(obj, node)
  -- Link line ins to SuperCollider
  for _, connection in ipairs(line_in_connections) do
    -- For each connection, look up the output and input Interest
    local out_port = interface_om:lookup(connection.output)
    local in_port = supercollider_om:lookup(connection.input)
    
    -- If they both exist, we can make the connection
    if out_port and in_port then
      link_ports(out_port, in_port)
    end
  end
  
  -- Link SuperCollider to line outs
  for _, connection in ipairs(line_out_connections) do
    -- For each connection, look up the output and input Interest
    local out_port = supercollider_om:lookup(connection.output)
    local in_port = interface_om:lookup(connection.input)
    
    -- If they both exist, we can make the connection
    if out_port and in_port then
      link_ports(out_port, in_port)
    end
  end
end

-- A debugger to verify that nodes are being detected by the ObjectManagers
local function debug(obj, node)
  print("Object added!")
  for k, v in pairs(node.properties) do
    print("  " .. tostring(k) .. " = " .. tostring(v))
  end
end

interface_om:connect("object-added", make_all_connections)
supercollider_om:connect("object-added", make_all_connections)

interface_om:activate()
supercollider_om:activate()


-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-- Additional functionality to destroy auto links to the internal sound card
-- This is necessary if we are using the Focusrite interface.
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

-- Tracks nodes representing the internal sound card
local ic_node_om = ObjectManager {
  Interest {
    type = "node",
    Constraint { "node.description", "matches", "Built-in Audio*" },
  },
}

-- Tracks nodes representing the Focusrite
local focusrite_node_om = ObjectManager {
  Interest {
    type = "node",
    Constraint { "node.description", "matches", "Scarlett*" },
  },
}

-- Tracks the SuperCollider node
local sc_node_om = ObjectManager {
  Interest {
    type = "node",
    Constraint { "node.description", "equals", "SuperCollider" },
  },
}

-- Tracks all links
local links_om = ObjectManager {
  Interest {
    type = "link",
  },
}

local function destroy_ic_links()
  if not focusrite_node_om:lookup() then return end
  for link in links_om:iterate() do
    local props = link.properties
    local link_in = props["link.input.node"]
    local link_out = props["link.output.node"]
  
    -- Check if we have a link from the integrated sound card microphone to SuperCollider
    local ic_card1 = ic_node_om:lookup(Interest {
      type = "node",
      Constraint { "object.id", "=", link_out },
    })
    local sc_1 = sc_node_om:lookup(Interest {
      type = "node",
      Constraint { "object.id", "=", link_in },
    })
    if ic_card1 and sc_1 then
      print("Found link from integrated sound card to SuperCollider. Requesting destroy...")
      link:request_destroy()
    elseif sc_1 then
      print("Found SuperCollider link in but no ic link.")
    end
  
    -- Check if we have a link from SuperCollider to the integrated sound card
    local sc_2 = sc_node_om:lookup(Interest {
      type = "node",
      Constraint { "object.id", "=", link_out },
    })
    local ic_card2 = ic_node_om:lookup(Interest {
      type = "node",
      Constraint { "object.id", "=", link_in },
    })
    if sc_2 and ic_card2 then
      print("Found link from SuperCollider to integrated sound card. Requesting destroy...")
      link:request_destroy()
    elseif sc_2 then
      print("Found SuperCollider link out but no ic link.")
    end
  end
end

ic_node_om:connect("object-added", destroy_ic_links)
sc_node_om:connect("object-added", destroy_ic_links)
focusrite_node_om:connect("object-added", destroy_ic_links)
links_om:connect("object-added", destroy_ic_links)
ic_node_om:activate()
sc_node_om:activate()
focusrite_node_om:activate()
links_om:activate()

print("Script focusrite-sc.lua loaded.")
