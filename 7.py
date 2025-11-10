def tower_of_hanoi(n,source,auxiliary,destination):
  if n == 1:
    print(f"move disk 1 from {source}->{destination}")
    return 1
  moves = 0
  moves += tower_of_hanoi(n-1,source,destination,auxiliary)
  print(f"moves disk {n} from {source}->{destination}")
  moves+=1
  moves += tower_of_hanoi(n-1,auxiliary,source,destination)
  return moves

n = int(input("enter number of disks: "))
print("\n Steps to solve tower_of_hanoi")
total_moves = tower_of_hanoi(n,"A","B","C")
print(f"\n✅ Total moves required: {total_moves}")
