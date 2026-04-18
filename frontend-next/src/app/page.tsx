import { DashboardPage } from "@/components/dashboard/dashboard-page";
import { getBlurDataURL } from "@/lib/images/get-blur-data-url";

export default async function HomePage() {
  const heroBlurDataURL = await getBlurDataURL("images/hero-lab.jpg");
  return <DashboardPage heroBlurDataURL={heroBlurDataURL} />;
}
