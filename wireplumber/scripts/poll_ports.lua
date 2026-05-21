port_om = ObjectManager {
  Interest {
    type = "port",
  },
}

port_om:connect("object-added", function(obj, node)
  print("Object added")
  for k, v in pairs(node.properties) do
    print("  " .. tostring(k) .. " = " .. tostring(v))
  end
end)

port_om:activate()

