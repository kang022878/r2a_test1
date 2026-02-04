import { FinancialHero } from "@/components/ui/hero-section"

const FinancialHeroDemo = () => {
  return (
    <div className="w-full bg-background">
      <FinancialHero
        title={
          <>
            Ready to Transform Your <br />
            <span className="text-primary">Management?</span>
          </>
        }
        description="Experience the future of finance with our cutting-edge SaaS platform. Start optimizing your financial operations today!"
        buttonText="Open Camera"
        buttonLink="/camera"
        imageUrl1="https://images.unsplash.com/photo-1579965342575-16428a7c8881?auto=format&fit=crop&w=900&q=60"
        imageUrl2="https://images.unsplash.com/photo-1664013263421-91e3a8101259?auto=format&fit=crop&w=900&q=60"
      />
    </div>
  )
}

export default FinancialHeroDemo
