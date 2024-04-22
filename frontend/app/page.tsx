"use client";

import { Player } from "@lottiefiles/react-lottie-player";
import { useState } from "react";
import { BloodLottie } from "@/assets";

export default function Home() {
  const [isTextInput, setIsTextInput] = useState(true)

  return (
    <main className="flex flex-1 flex-col w-[100vw] h-[100vh] bg-[#faf8f8]">
      <h1 className="text-4xl font-bold font-mono m-8">To Kill a Mocking Text 🩸</h1>
      <div className="m-8 mt-0">
        <button onClick={() => setIsTextInput(true)} className={`text-xl font-mono px-4 py-1 border-2 rounded-xl mr-2  transition-all duration-200 ${isTextInput ? "border-red-500 text-red-500" : "border-slate-500 text-slate-500"}`}>Text</button>
        <button onClick={() => setIsTextInput(false)} className={`text-xl font-mono px-4 py-1 border-2 rounded-xl mr-2 transition-all duration-200 ${!isTextInput ? "border-red-500 text-red-500" : "border-slate-500 text-slate-500"}`}>File</button>
      </div>
      <div className="flex flex-row">
        <div>
          <textarea
            placeholder="Your paper abstract here ..."
            className="border-2 border-red-500 rounded-xl w-[45vw] h-[72vh] mx-8 placeholder:font-mono font-mono outline-none focus:outline-none focus:ring-2 focus:ring-red-500"
          />
        </div>
        <div className="flex flex-col items-center justify-center align-middle w-full">
          <Player autoplay
            loop
            src={BloodLottie}
            style={{ height: '20rem', width: '20rem' }} />
        </div>
      </div>
    </main>
  );
}
