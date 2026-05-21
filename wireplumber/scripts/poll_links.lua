-- This script watches all links that are created,
-- and prints useful information, including the node and port names.

port_om = ObjectManager {
  Interest {
    type = "port",
  },
}

link_om = ObjectManager {
  Interest {
    type = "link",
  },
}

link_om:connect("object-added", function(obj, link)
  -- Retrieve the node names
  local props = link.properties
  local port_out = port_om:lookup(Interest {
    type = "port",
    Constraint { "object.id", "=", props["link.output.port"] },
  })
  local port_in = port_om:lookup(Interest {
    type = "port",
    Constraint { "object.id", "=", props["link.input.port"] },
  })
  local port_out_alias = port_out and port_out.properties["port.alias"] or "unknown"
  local port_in_alias = port_in and port_in.properties["port.alias"] or "unknown"
  
  print("Link created")
  print("  From " .. port_out_alias)
  print("  To " .. port_in_alias)
  print("  Link detail:")
  for k, v in pairs(link.properties) do
    print("  " .. tostring(k) .. " = " .. tostring(v))
  end  
end)

port_om:activate()
link_om:activate()

