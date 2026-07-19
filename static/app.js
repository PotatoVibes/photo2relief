async function checkHealth() {
  const res = await fetch("/api/health");
  const body = await res.json();
  console.log("photo2relief health:", body);
}

checkHealth();
