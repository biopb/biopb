import { Outlet } from "react-router-dom";
import { ClientBootstrap } from "./ClientBootstrap";

// The viewer/admin/unlock surfaces share the tensor-data-plane bootstrap (token
// gate + readyz wait + source load); the dashboard and observe pages talk to the
// control / session APIs directly and need none of it, so ClientBootstrap wraps
// only this route group.
export function ViewerLayout() {
  return (
    <>
      <ClientBootstrap />
      <Outlet />
    </>
  );
}
