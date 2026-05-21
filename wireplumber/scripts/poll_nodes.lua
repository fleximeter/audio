node_om = ObjectManager {
  Interest {
    type = "node",
  },
}

node_om:connect("object-added", function(obj, node)
  print("Node added")
  for k, v in pairs(node.properties) do
    print("  " .. tostring(k) .. " = " .. tostring(v))
  end
end)

node_om:activate()

